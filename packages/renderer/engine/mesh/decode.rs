use super::*;

pub fn decode_texture(
    data: &[u8],
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
) -> Result<PreparedTexture> {
    decode_texture_with_label(
        data,
        width_hint,
        height_hint,
        &format!("scene.meshes[{mesh_index}].texture"),
    )
}

pub fn decode_texture_with_label(
    data: &[u8],
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    label: &str,
) -> Result<PreparedTexture> {
    let w = width_hint.unwrap_or(0);
    let h = height_hint.unwrap_or(0);

    if w > 0 && h > 0 && data.len() >= (w as usize) * (h as usize) * 4 {
        let base_len = (w as usize) * (h as usize) * 4;
        let mut mipmaps = Vec::new();
        let mut offset = base_len;
        let mut mip_width = w;
        let mut mip_height = h;
        let mut level = 1u32;

        while offset < data.len() {
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);
            let level_len = (mip_width as usize) * (mip_height as usize) * 4;
            if data.len() - offset < level_len {
                bail!(
                    "{label}: explicit mipmap level {level} expected {mip_width}x{mip_height} RGBA8 bytes"
                );
            }
            let level_end = offset + level_len;
            mipmaps.push(PreparedTextureMipLevel {
                rgba: data[offset..level_end].to_vec(),
                width: mip_width,
                height: mip_height,
            });
            offset = level_end;
            if mip_width == 1 && mip_height == 1 {
                if offset != data.len() {
                    bail!("{label}: explicit mipmap data continues after the 1x1 level");
                }
                break;
            }
            level += 1;
        }
        if !mipmaps.is_empty() && (mip_width != 1 || mip_height != 1) {
            bail!("{label}: explicit mipmap data must include the complete chain down to 1x1");
        }

        return Ok(PreparedTexture {
            rgba: data[..base_len].to_vec(),
            width: w,
            height: h,
            mipmaps,
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::None,
            anisotropy: 1,
        });
    }

    let img = image::load_from_memory(data)
        .with_context(|| format!("{label}: failed to decode image ({} bytes)", data.len()))?;
    let rgba = img.to_rgba8();
    Ok(PreparedTexture {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
        mipmaps: Vec::new(),
        wrap_s: WrapMode::ClampToEdge,
        wrap_t: WrapMode::ClampToEdge,
        mag_filter: TextureFilter::Linear,
        min_filter: TextureFilter::Linear,
        mipmap_filter: MipmapFilter::None,
        anisotropy: 1,
    })
}

pub(super) fn decode_optional_texture(
    data: Option<&[u8]>,
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
) -> Result<Option<PreparedTexture>> {
    match data {
        Some(tex_data) if !tex_data.is_empty() => Ok(Some(decode_texture(
            tex_data,
            width_hint,
            height_hint,
            mesh_index,
        )?)),
        _ => Ok(None),
    }
}

pub(super) fn decode_optional_texture_with_sampling(
    data: Option<&[u8]>,
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
    wrap_s: Option<&str>,
    wrap_t: Option<&str>,
    mag_filter: Option<&str>,
    min_filter: Option<&str>,
    anisotropy: Option<f64>,
    anisotropy_field: &str,
) -> Result<Option<PreparedTexture>> {
    let mut texture = decode_optional_texture(data, width_hint, height_hint, mesh_index)?;
    if let Some(tex) = texture.as_mut() {
        apply_texture_sampling(
            tex,
            wrap_s,
            wrap_t,
            mag_filter,
            min_filter,
            anisotropy,
            anisotropy_field,
        )?;
    }
    Ok(texture)
}

pub(super) fn apply_texture_sampling(
    texture: &mut PreparedTexture,
    wrap_s: Option<&str>,
    wrap_t: Option<&str>,
    mag_filter: Option<&str>,
    min_filter: Option<&str>,
    anisotropy: Option<f64>,
    anisotropy_field: &str,
) -> Result<()> {
    texture.wrap_s = WrapMode::from_str_opt(wrap_s);
    texture.wrap_t = WrapMode::from_str_opt(wrap_t);
    texture.mag_filter = TextureFilter::from_str_opt(mag_filter);
    texture.min_filter = TextureFilter::from_min_filter_str(min_filter);
    texture.mipmap_filter = MipmapFilter::from_min_filter_str(min_filter);
    texture.anisotropy = texture_anisotropy(anisotropy, anisotropy_field)?;
    Ok(())
}

pub(super) struct SurfaceTextureInputs {
    pub(super) texture: Option<PreparedTexture>,
    pub(super) texture_transform: [f32; 6],
    pub(super) alpha_map_transform: [f32; 6],
    pub(super) normal_map_transform: [f32; 6],
    pub(super) bump_map_transform: [f32; 6],
    pub(super) matcap_map_transform: [f32; 6],
    pub(super) displacement_map_transform: [f32; 6],
    pub(super) displacement_map_is_srgb: bool,
    pub(super) metallic_roughness_texture_transform: [f32; 6],
    pub(super) emissive_map_transform: [f32; 6],
    pub(super) ao_map_transform: [f32; 6],
    pub(super) light_map_transform: [f32; 6],
    pub(super) specular_map_transform: [f32; 6],
    pub(super) specular_color_map_transform: [f32; 6],
    pub(super) specular_intensity_map_transform: [f32; 6],
    pub(super) clearcoat_map_transform: [f32; 6],
    pub(super) clearcoat_roughness_map_transform: [f32; 6],
    pub(super) clearcoat_normal_map_transform: [f32; 6],
    pub(super) sheen_color_map_transform: [f32; 6],
    pub(super) sheen_roughness_map_transform: [f32; 6],
    pub(super) anisotropy_map_transform: [f32; 6],
    pub(super) iridescence_map_transform: [f32; 6],
    pub(super) iridescence_thickness_map_transform: [f32; 6],
    pub(super) transmission_map_transform: [f32; 6],
    pub(super) thickness_map_transform: [f32; 6],
    pub(super) texture_is_srgb: bool,
    pub(super) alpha_map_is_srgb: bool,
    pub(super) matcap_map_is_srgb: bool,
    pub(super) gradient_map_is_srgb: bool,
    pub(super) metallic_roughness_texture_is_srgb: bool,
    pub(super) emissive_map_is_srgb: bool,
    pub(super) ao_map_is_srgb: bool,
    pub(super) light_map_is_srgb: bool,
    pub(super) specular_map_is_srgb: bool,
    pub(super) common: CommonTextureInputs,
}

pub(super) fn prepare_surface_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<SurfaceTextureInputs> {
    thread::scope(|scope| {
        let texture = has_texture_data(mesh.texture.as_deref()).then(|| {
            scope.spawn(|| {
                decode_optional_texture_with_sampling(
                    mesh.texture.as_deref(),
                    mesh.texture_width,
                    mesh.texture_height,
                    mesh_index,
                    mesh.texture_wrap_s.as_deref(),
                    mesh.texture_wrap_t.as_deref(),
                    mesh.texture_mag_filter.as_deref(),
                    mesh.texture_min_filter.as_deref(),
                    mesh.texture_anisotropy,
                    &format!("scene.meshes[{mesh_index}].textureAnisotropy"),
                )
            })
        });
        let alpha_map = has_texture_data(mesh.alpha_map.as_deref()).then(|| {
            scope.spawn(|| {
                decode_optional_texture_with_sampling(
                    mesh.alpha_map.as_deref(),
                    mesh.alpha_map_width,
                    mesh.alpha_map_height,
                    mesh_index,
                    mesh.alpha_map_wrap_s.as_deref(),
                    mesh.alpha_map_wrap_t.as_deref(),
                    mesh.alpha_map_mag_filter.as_deref(),
                    mesh.alpha_map_min_filter.as_deref(),
                    mesh.alpha_map_anisotropy,
                    &format!("scene.meshes[{mesh_index}].alphaMapAnisotropy"),
                )
            })
        });

        let texture = match texture {
            Some(texture) => join_texture_worker(texture, "base texture worker")?,
            None => None,
        };

        let texture_transform =
            parse_texture_transform(mesh.texture_transform.as_deref(), mesh_index)?;
        let alpha_map_transform =
            parse_texture_transform(mesh.alpha_map_transform.as_deref(), mesh_index)?;
        let normal_map_transform =
            parse_texture_transform(mesh.normal_map_transform.as_deref(), mesh_index)?;
        let bump_map_transform =
            parse_texture_transform(mesh.bump_map_transform.as_deref(), mesh_index)?;
        let matcap_map_transform =
            parse_texture_transform(mesh.matcap_map_transform.as_deref(), mesh_index)?;
        let displacement_map_transform =
            parse_texture_transform(mesh.displacement_map_transform.as_deref(), mesh_index)?;
        let metallic_roughness_texture_transform = parse_texture_transform(
            mesh.metallic_roughness_texture_transform.as_deref(),
            mesh_index,
        )?;
        let emissive_map_transform =
            parse_texture_transform(mesh.emissive_map_transform.as_deref(), mesh_index)?;
        let ao_map_transform =
            parse_texture_transform(mesh.ao_map_transform.as_deref(), mesh_index)?;
        let light_map_transform =
            parse_texture_transform(mesh.light_map_transform.as_deref(), mesh_index)?;
        let specular_map_transform =
            parse_texture_transform(mesh.specular_map_transform.as_deref(), mesh_index)?;
        let specular_color_map_transform =
            parse_texture_transform(mesh.specular_color_map_transform.as_deref(), mesh_index)?;
        let specular_intensity_map_transform =
            parse_texture_transform(mesh.specular_intensity_map_transform.as_deref(), mesh_index)?;
        let clearcoat_map_transform =
            parse_texture_transform(mesh.clearcoat_map_transform.as_deref(), mesh_index)?;
        let clearcoat_roughness_map_transform = parse_texture_transform(
            mesh.clearcoat_roughness_map_transform.as_deref(),
            mesh_index,
        )?;
        let clearcoat_normal_map_transform =
            parse_texture_transform(mesh.clearcoat_normal_map_transform.as_deref(), mesh_index)?;
        let sheen_color_map_transform =
            parse_texture_transform(mesh.sheen_color_map_transform.as_deref(), mesh_index)?;
        let sheen_roughness_map_transform =
            parse_texture_transform(mesh.sheen_roughness_map_transform.as_deref(), mesh_index)?;
        let anisotropy_map_transform =
            parse_texture_transform(mesh.anisotropy_map_transform.as_deref(), mesh_index)?;
        let iridescence_map_transform =
            parse_texture_transform(mesh.iridescence_map_transform.as_deref(), mesh_index)?;
        let iridescence_thickness_map_transform = parse_texture_transform(
            mesh.iridescence_thickness_map_transform.as_deref(),
            mesh_index,
        )?;
        let transmission_map_transform =
            parse_texture_transform(mesh.transmission_map_transform.as_deref(), mesh_index)?;
        let thickness_map_transform =
            parse_texture_transform(mesh.thickness_map_transform.as_deref(), mesh_index)?;
        let texture_is_srgb = matches!(mesh.texture_color_space.as_deref(), Some("srgb"));
        let alpha_map_is_srgb = matches!(mesh.alpha_map_color_space.as_deref(), Some("srgb"));
        let matcap_map_is_srgb = matches!(mesh.matcap_map_color_space.as_deref(), Some("srgb"));
        let gradient_map_is_srgb = matches!(mesh.gradient_map_color_space.as_deref(), Some("srgb"));
        let metallic_roughness_texture_is_srgb = matches!(
            mesh.metallic_roughness_texture_color_space.as_deref(),
            Some("srgb")
        );
        let emissive_map_is_srgb = matches!(mesh.emissive_map_color_space.as_deref(), Some("srgb"));
        let ao_map_is_srgb = matches!(mesh.ao_map_color_space.as_deref(), Some("srgb"));
        let light_map_is_srgb = matches!(mesh.light_map_color_space.as_deref(), Some("srgb"));
        let specular_map_is_srgb = matches!(mesh.specular_map_color_space.as_deref(), Some("srgb"));
        let displacement_map_is_srgb =
            matches!(mesh.displacement_map_color_space.as_deref(), Some("srgb"));

        let mut common = prepare_common_texture_inputs(mesh, mesh_index)?;
        common.alpha_map = match alpha_map {
            Some(alpha_map) => join_texture_worker(alpha_map, "alpha map worker")?,
            None => None,
        };

        Ok(SurfaceTextureInputs {
            texture,
            texture_transform,
            alpha_map_transform,
            normal_map_transform,
            bump_map_transform,
            matcap_map_transform,
            displacement_map_transform,
            displacement_map_is_srgb,
            metallic_roughness_texture_transform,
            emissive_map_transform,
            ao_map_transform,
            light_map_transform,
            specular_map_transform,
            specular_color_map_transform,
            specular_intensity_map_transform,
            clearcoat_map_transform,
            clearcoat_roughness_map_transform,
            clearcoat_normal_map_transform,
            sheen_color_map_transform,
            sheen_roughness_map_transform,
            anisotropy_map_transform,
            iridescence_map_transform,
            iridescence_thickness_map_transform,
            transmission_map_transform,
            thickness_map_transform,
            texture_is_srgb,
            alpha_map_is_srgb,
            matcap_map_is_srgb,
            gradient_map_is_srgb,
            metallic_roughness_texture_is_srgb,
            emissive_map_is_srgb,
            ao_map_is_srgb,
            light_map_is_srgb,
            specular_map_is_srgb,
            common,
        })
    })
}
