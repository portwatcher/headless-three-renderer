use super::*;

pub(super) fn has_texture_data(data: Option<&[u8]>) -> bool {
    data.is_some_and(|data| !data.is_empty())
}

#[derive(Default)]
pub(super) struct CommonTextureInputs {
    pub(super) normal_map: Option<PreparedTexture>,
    pub(super) normal_map_type: NormalMapType,
    pub(super) normal_scale: [f32; 2],
    pub(super) bump_map: Option<PreparedTexture>,
    pub(super) bump_scale: f32,
    pub(super) displacement_map: Option<PreparedTexture>,
    pub(super) displacement_scale: f32,
    pub(super) displacement_bias: f32,
    pub(super) gradient_map: Option<PreparedTexture>,
    pub(super) matcap_map: Option<PreparedTexture>,
    pub(super) metallic_roughness_texture: Option<PreparedTexture>,
    pub(super) specular_map: Option<PreparedTexture>,
    pub(super) emissive_map: Option<PreparedTexture>,
    pub(super) ao_map: Option<PreparedTexture>,
    pub(super) ao_map_intensity: f32,
    pub(super) light_map: Option<PreparedTexture>,
    pub(super) light_map_intensity: f32,
    pub(super) alpha_map: Option<PreparedTexture>,
}

pub(super) fn prepare_common_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<CommonTextureInputs> {
    if !has_common_texture_input(mesh) {
        return Ok(CommonTextureInputs {
            normal_map_type: NormalMapType::from_str_opt(
                mesh.normal_map_type.as_deref(),
                mesh_index,
            )?,
            normal_scale: parse_optional_vec2(
                mesh.normal_scale.as_deref(),
                [1.0, 1.0],
                "mesh normalScale",
            )?,
            bump_scale: finite_f32(mesh.bump_scale.unwrap_or(1.0), "mesh bumpScale")?,
            displacement_scale: finite_f32(
                mesh.displacement_scale.unwrap_or(1.0),
                "mesh displacementScale",
            )?,
            displacement_bias: finite_f32(
                mesh.displacement_bias.unwrap_or(0.0),
                "mesh displacementBias",
            )?,
            ao_map_intensity: finite_clamp01(
                mesh.ao_map_intensity.unwrap_or(1.0),
                "mesh aoMapIntensity",
            )?,
            light_map_intensity: finite_f32(
                mesh.light_map_intensity.unwrap_or(1.0),
                "mesh lightMapIntensity",
            )?
            .max(0.0),
            ..CommonTextureInputs::default()
        });
    }

    thread::scope(|scope| {
        let normal_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.normal_map.as_deref(),
                mesh.normal_map_width,
                mesh.normal_map_height,
                mesh_index,
                mesh.normal_map_wrap_s.as_deref(),
                mesh.normal_map_wrap_t.as_deref(),
                mesh.normal_map_mag_filter.as_deref(),
                mesh.normal_map_min_filter.as_deref(),
                mesh.normal_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].normalMapAnisotropy"),
            )
        });
        let bump_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.bump_map.as_deref(),
                mesh.bump_map_width,
                mesh.bump_map_height,
                mesh_index,
                mesh.bump_map_wrap_s.as_deref(),
                mesh.bump_map_wrap_t.as_deref(),
                mesh.bump_map_mag_filter.as_deref(),
                mesh.bump_map_min_filter.as_deref(),
                mesh.bump_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].bumpMapAnisotropy"),
            )
        });
        let displacement_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.displacement_map.as_deref(),
                mesh.displacement_map_width,
                mesh.displacement_map_height,
                mesh_index,
                mesh.displacement_map_wrap_s.as_deref(),
                mesh.displacement_map_wrap_t.as_deref(),
                mesh.displacement_map_mag_filter.as_deref(),
                mesh.displacement_map_min_filter.as_deref(),
                mesh.displacement_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].displacementMapAnisotropy"),
            )
        });
        let gradient_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.gradient_map.as_deref(),
                mesh.gradient_map_width,
                mesh.gradient_map_height,
                mesh_index,
                mesh.gradient_map_wrap_s.as_deref(),
                mesh.gradient_map_wrap_t.as_deref(),
                mesh.gradient_map_mag_filter.as_deref(),
                mesh.gradient_map_min_filter.as_deref(),
                mesh.gradient_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].gradientMapAnisotropy"),
            )
        });
        let matcap_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.matcap_map.as_deref(),
                mesh.matcap_map_width,
                mesh.matcap_map_height,
                mesh_index,
                mesh.matcap_map_wrap_s.as_deref(),
                mesh.matcap_map_wrap_t.as_deref(),
                mesh.matcap_map_mag_filter.as_deref(),
                mesh.matcap_map_min_filter.as_deref(),
                mesh.matcap_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].matcapMapAnisotropy"),
            )
        });
        let metallic_roughness_texture = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.metallic_roughness_texture.as_deref(),
                mesh.metallic_roughness_texture_width,
                mesh.metallic_roughness_texture_height,
                mesh_index,
                mesh.metallic_roughness_texture_wrap_s.as_deref(),
                mesh.metallic_roughness_texture_wrap_t.as_deref(),
                mesh.metallic_roughness_texture_mag_filter.as_deref(),
                mesh.metallic_roughness_texture_min_filter.as_deref(),
                mesh.metallic_roughness_texture_anisotropy,
                &format!("scene.meshes[{mesh_index}].metallicRoughnessTextureAnisotropy"),
            )
        });
        let specular_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_map.as_deref(),
                mesh.specular_map_width,
                mesh.specular_map_height,
                mesh_index,
                mesh.specular_map_wrap_s.as_deref(),
                mesh.specular_map_wrap_t.as_deref(),
                mesh.specular_map_mag_filter.as_deref(),
                mesh.specular_map_min_filter.as_deref(),
                mesh.specular_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularMapAnisotropy"),
            )
        });
        let emissive_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.emissive_map.as_deref(),
                mesh.emissive_map_width,
                mesh.emissive_map_height,
                mesh_index,
                mesh.emissive_map_wrap_s.as_deref(),
                mesh.emissive_map_wrap_t.as_deref(),
                mesh.emissive_map_mag_filter.as_deref(),
                mesh.emissive_map_min_filter.as_deref(),
                mesh.emissive_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].emissiveMapAnisotropy"),
            )
        });
        let ao_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.ao_map.as_deref(),
                mesh.ao_map_width,
                mesh.ao_map_height,
                mesh_index,
                mesh.ao_map_wrap_s.as_deref(),
                mesh.ao_map_wrap_t.as_deref(),
                mesh.ao_map_mag_filter.as_deref(),
                mesh.ao_map_min_filter.as_deref(),
                mesh.ao_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].aoMapAnisotropy"),
            )
        });
        let light_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.light_map.as_deref(),
                mesh.light_map_width,
                mesh.light_map_height,
                mesh_index,
                mesh.light_map_wrap_s.as_deref(),
                mesh.light_map_wrap_t.as_deref(),
                mesh.light_map_mag_filter.as_deref(),
                mesh.light_map_min_filter.as_deref(),
                mesh.light_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].lightMapAnisotropy"),
            )
        });
        let mut normal_map = join_texture_worker(normal_map, "normal map worker")?;
        if matches!(mesh.normal_map_color_space.as_deref(), Some("srgb")) {
            if let Some(texture) = normal_map.as_mut() {
                decode_texture_rgb_srgb_to_linear(texture);
            }
        }
        let normal_map_type =
            NormalMapType::from_str_opt(mesh.normal_map_type.as_deref(), mesh_index)?;
        let normal_scale =
            parse_optional_vec2(mesh.normal_scale.as_deref(), [1.0, 1.0], "mesh normalScale")?;
        let mut bump_map = join_texture_worker(bump_map, "bump map worker")?;
        if matches!(mesh.bump_map_color_space.as_deref(), Some("srgb")) {
            if let Some(texture) = bump_map.as_mut() {
                decode_texture_rgb_srgb_to_linear(texture);
            }
        }
        let bump_scale = finite_f32(mesh.bump_scale.unwrap_or(1.0), "mesh bumpScale")?;
        let displacement_map = join_texture_worker(displacement_map, "displacement map worker")?;
        let displacement_scale = finite_f32(
            mesh.displacement_scale.unwrap_or(1.0),
            "mesh displacementScale",
        )?;
        let displacement_bias = finite_f32(
            mesh.displacement_bias.unwrap_or(0.0),
            "mesh displacementBias",
        )?;
        let gradient_map = join_texture_worker(gradient_map, "gradient map worker")?;
        let matcap_map = join_texture_worker(matcap_map, "matcap map worker")?;
        let metallic_roughness_texture = join_texture_worker(
            metallic_roughness_texture,
            "metallic roughness texture worker",
        )?;
        let specular_map = join_texture_worker(specular_map, "specular map worker")?;
        let emissive_map = join_texture_worker(emissive_map, "emissive map worker")?;
        let ao_map = join_texture_worker(ao_map, "ao map worker")?;
        let ao_map_intensity =
            finite_clamp01(mesh.ao_map_intensity.unwrap_or(1.0), "mesh aoMapIntensity")?;
        let light_map = join_texture_worker(light_map, "light map worker")?;
        let light_map_intensity = finite_f32(
            mesh.light_map_intensity.unwrap_or(1.0),
            "mesh lightMapIntensity",
        )?
        .max(0.0);

        Ok(CommonTextureInputs {
            normal_map,
            normal_map_type,
            normal_scale,
            bump_map,
            bump_scale,
            displacement_map,
            displacement_scale,
            displacement_bias,
            gradient_map,
            matcap_map,
            metallic_roughness_texture,
            specular_map,
            emissive_map,
            ao_map,
            ao_map_intensity,
            light_map,
            light_map_intensity,
            alpha_map: None,
        })
    })
}

pub(super) fn has_common_texture_input(mesh: &SceneMesh) -> bool {
    [
        mesh.normal_map.as_deref(),
        mesh.bump_map.as_deref(),
        mesh.displacement_map.as_deref(),
        mesh.gradient_map.as_deref(),
        mesh.matcap_map.as_deref(),
        mesh.metallic_roughness_texture.as_deref(),
        mesh.specular_map.as_deref(),
        mesh.emissive_map.as_deref(),
        mesh.ao_map.as_deref(),
        mesh.light_map.as_deref(),
    ]
    .into_iter()
    .any(has_texture_data)
}

#[derive(Default)]
pub(super) struct PhysicalTextureInputs {
    pub(super) clearcoat_map: Option<PreparedTexture>,
    pub(super) clearcoat_roughness_map: Option<PreparedTexture>,
    pub(super) clearcoat_normal_map: Option<PreparedTexture>,
    pub(super) sheen_color_map: Option<PreparedTexture>,
    pub(super) sheen_roughness_map: Option<PreparedTexture>,
    pub(super) anisotropy_map: Option<PreparedTexture>,
    pub(super) iridescence_map: Option<PreparedTexture>,
    pub(super) iridescence_thickness_map: Option<PreparedTexture>,
    pub(super) transmission_map: Option<PreparedTexture>,
    pub(super) thickness_map: Option<PreparedTexture>,
    pub(super) specular_color_map: Option<PreparedTexture>,
    pub(super) specular_intensity_map: Option<PreparedTexture>,
}

pub(super) fn prepare_physical_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<PhysicalTextureInputs> {
    if !has_physical_texture_input(mesh) {
        return Ok(PhysicalTextureInputs::default());
    }

    thread::scope(|scope| {
        let clearcoat_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_map.as_deref(),
                mesh.clearcoat_map_width,
                mesh.clearcoat_map_height,
                mesh_index,
                mesh.clearcoat_map_wrap_s.as_deref(),
                mesh.clearcoat_map_wrap_t.as_deref(),
                mesh.clearcoat_map_mag_filter.as_deref(),
                mesh.clearcoat_map_min_filter.as_deref(),
                mesh.clearcoat_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatMapAnisotropy"),
            )
        });
        let clearcoat_roughness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_roughness_map.as_deref(),
                mesh.clearcoat_roughness_map_width,
                mesh.clearcoat_roughness_map_height,
                mesh_index,
                mesh.clearcoat_roughness_map_wrap_s.as_deref(),
                mesh.clearcoat_roughness_map_wrap_t.as_deref(),
                mesh.clearcoat_roughness_map_mag_filter.as_deref(),
                mesh.clearcoat_roughness_map_min_filter.as_deref(),
                mesh.clearcoat_roughness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatRoughnessMapAnisotropy"),
            )
        });
        let clearcoat_normal_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_normal_map.as_deref(),
                mesh.clearcoat_normal_map_width,
                mesh.clearcoat_normal_map_height,
                mesh_index,
                mesh.clearcoat_normal_map_wrap_s.as_deref(),
                mesh.clearcoat_normal_map_wrap_t.as_deref(),
                mesh.clearcoat_normal_map_mag_filter.as_deref(),
                mesh.clearcoat_normal_map_min_filter.as_deref(),
                mesh.clearcoat_normal_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatNormalMapAnisotropy"),
            )
        });
        let sheen_color_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.sheen_color_map.as_deref(),
                mesh.sheen_color_map_width,
                mesh.sheen_color_map_height,
                mesh_index,
                mesh.sheen_color_map_wrap_s.as_deref(),
                mesh.sheen_color_map_wrap_t.as_deref(),
                mesh.sheen_color_map_mag_filter.as_deref(),
                mesh.sheen_color_map_min_filter.as_deref(),
                mesh.sheen_color_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].sheenColorMapAnisotropy"),
            )
        });
        let sheen_roughness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.sheen_roughness_map.as_deref(),
                mesh.sheen_roughness_map_width,
                mesh.sheen_roughness_map_height,
                mesh_index,
                mesh.sheen_roughness_map_wrap_s.as_deref(),
                mesh.sheen_roughness_map_wrap_t.as_deref(),
                mesh.sheen_roughness_map_mag_filter.as_deref(),
                mesh.sheen_roughness_map_min_filter.as_deref(),
                mesh.sheen_roughness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].sheenRoughnessMapAnisotropy"),
            )
        });
        let anisotropy_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.anisotropy_map.as_deref(),
                mesh.anisotropy_map_width,
                mesh.anisotropy_map_height,
                mesh_index,
                mesh.anisotropy_map_wrap_s.as_deref(),
                mesh.anisotropy_map_wrap_t.as_deref(),
                mesh.anisotropy_map_mag_filter.as_deref(),
                mesh.anisotropy_map_min_filter.as_deref(),
                mesh.anisotropy_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].anisotropyMapAnisotropy"),
            )
        });
        let iridescence_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.iridescence_map.as_deref(),
                mesh.iridescence_map_width,
                mesh.iridescence_map_height,
                mesh_index,
                mesh.iridescence_map_wrap_s.as_deref(),
                mesh.iridescence_map_wrap_t.as_deref(),
                mesh.iridescence_map_mag_filter.as_deref(),
                mesh.iridescence_map_min_filter.as_deref(),
                mesh.iridescence_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].iridescenceMapAnisotropy"),
            )
        });
        let iridescence_thickness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.iridescence_thickness_map.as_deref(),
                mesh.iridescence_thickness_map_width,
                mesh.iridescence_thickness_map_height,
                mesh_index,
                mesh.iridescence_thickness_map_wrap_s.as_deref(),
                mesh.iridescence_thickness_map_wrap_t.as_deref(),
                mesh.iridescence_thickness_map_mag_filter.as_deref(),
                mesh.iridescence_thickness_map_min_filter.as_deref(),
                mesh.iridescence_thickness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].iridescenceThicknessMapAnisotropy"),
            )
        });
        let transmission_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.transmission_map.as_deref(),
                mesh.transmission_map_width,
                mesh.transmission_map_height,
                mesh_index,
                mesh.transmission_map_wrap_s.as_deref(),
                mesh.transmission_map_wrap_t.as_deref(),
                mesh.transmission_map_mag_filter.as_deref(),
                mesh.transmission_map_min_filter.as_deref(),
                mesh.transmission_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].transmissionMapAnisotropy"),
            )
        });
        let thickness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.thickness_map.as_deref(),
                mesh.thickness_map_width,
                mesh.thickness_map_height,
                mesh_index,
                mesh.thickness_map_wrap_s.as_deref(),
                mesh.thickness_map_wrap_t.as_deref(),
                mesh.thickness_map_mag_filter.as_deref(),
                mesh.thickness_map_min_filter.as_deref(),
                mesh.thickness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].thicknessMapAnisotropy"),
            )
        });
        let specular_color_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_color_map.as_deref(),
                mesh.specular_color_map_width,
                mesh.specular_color_map_height,
                mesh_index,
                mesh.specular_color_map_wrap_s.as_deref(),
                mesh.specular_color_map_wrap_t.as_deref(),
                mesh.specular_color_map_mag_filter.as_deref(),
                mesh.specular_color_map_min_filter.as_deref(),
                mesh.specular_color_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularColorMapAnisotropy"),
            )
        });
        let specular_intensity_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_intensity_map.as_deref(),
                mesh.specular_intensity_map_width,
                mesh.specular_intensity_map_height,
                mesh_index,
                mesh.specular_intensity_map_wrap_s.as_deref(),
                mesh.specular_intensity_map_wrap_t.as_deref(),
                mesh.specular_intensity_map_mag_filter.as_deref(),
                mesh.specular_intensity_map_min_filter.as_deref(),
                mesh.specular_intensity_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularIntensityMapAnisotropy"),
            )
        });

        Ok(PhysicalTextureInputs {
            clearcoat_map: join_texture_worker(clearcoat_map, "clearcoat map worker")?,
            clearcoat_roughness_map: join_texture_worker(
                clearcoat_roughness_map,
                "clearcoat roughness map worker",
            )?,
            clearcoat_normal_map: {
                let mut texture =
                    join_texture_worker(clearcoat_normal_map, "clearcoat normal map worker")?;
                if matches!(
                    mesh.clearcoat_normal_map_color_space.as_deref(),
                    Some("srgb")
                ) {
                    if let Some(texture) = texture.as_mut() {
                        decode_texture_rgb_srgb_to_linear(texture);
                    }
                }
                texture
            },
            sheen_color_map: join_texture_worker(sheen_color_map, "sheen color map worker")?,
            sheen_roughness_map: join_texture_worker(
                sheen_roughness_map,
                "sheen roughness map worker",
            )?,
            anisotropy_map: join_texture_worker(anisotropy_map, "anisotropy map worker")?,
            iridescence_map: join_texture_worker(iridescence_map, "iridescence map worker")?,
            iridescence_thickness_map: join_texture_worker(
                iridescence_thickness_map,
                "iridescence thickness map worker",
            )?,
            transmission_map: join_texture_worker(transmission_map, "transmission map worker")?,
            thickness_map: join_texture_worker(thickness_map, "thickness map worker")?,
            specular_color_map: join_texture_worker(
                specular_color_map,
                "specular color map worker",
            )?,
            specular_intensity_map: join_texture_worker(
                specular_intensity_map,
                "specular intensity map worker",
            )?,
        })
    })
}

pub(super) fn has_physical_texture_input(mesh: &SceneMesh) -> bool {
    [
        mesh.clearcoat_map.as_deref(),
        mesh.clearcoat_roughness_map.as_deref(),
        mesh.clearcoat_normal_map.as_deref(),
        mesh.sheen_color_map.as_deref(),
        mesh.sheen_roughness_map.as_deref(),
        mesh.anisotropy_map.as_deref(),
        mesh.iridescence_map.as_deref(),
        mesh.iridescence_thickness_map.as_deref(),
        mesh.transmission_map.as_deref(),
        mesh.thickness_map.as_deref(),
        mesh.specular_color_map.as_deref(),
        mesh.specular_intensity_map.as_deref(),
    ]
    .into_iter()
    .any(|data| data.is_some_and(|data| !data.is_empty()))
}
