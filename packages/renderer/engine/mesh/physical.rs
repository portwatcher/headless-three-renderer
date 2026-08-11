use super::*;

pub(super) fn join_texture_worker<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T>>,
    label: &str,
) -> Result<T> {
    handle.join().unwrap_or_else(|_| panic!("{label} panicked"))
}

pub(super) struct PhysicalMapInputs<'a> {
    pub(super) clearcoat: Option<&'a PreparedTexture>,
    pub(super) clearcoat_roughness: Option<&'a PreparedTexture>,
    pub(super) sheen_color: Option<&'a PreparedTexture>,
    pub(super) sheen_roughness: Option<&'a PreparedTexture>,
    pub(super) anisotropy: Option<&'a PreparedTexture>,
    pub(super) iridescence: Option<&'a PreparedTexture>,
    pub(super) iridescence_thickness: Option<&'a PreparedTexture>,
    pub(super) transmission: Option<&'a PreparedTexture>,
    pub(super) thickness: Option<&'a PreparedTexture>,
    pub(super) specular_color: Option<&'a PreparedTexture>,
    pub(super) specular_intensity: Option<&'a PreparedTexture>,
    pub(super) clearcoat_is_srgb: bool,
    pub(super) clearcoat_roughness_is_srgb: bool,
    pub(super) anisotropy_is_srgb: bool,
    pub(super) iridescence_is_srgb: bool,
    pub(super) iridescence_thickness_is_srgb: bool,
    pub(super) transmission_is_srgb: bool,
    pub(super) thickness_is_srgb: bool,
    pub(super) sheen_color_is_srgb: bool,
    pub(super) specular_color_is_srgb: bool,
}

pub(super) fn pack_physical_maps(inputs: PhysicalMapInputs<'_>) -> Option<PreparedPhysicalMaps> {
    let maps = [
        inputs.clearcoat,
        inputs.clearcoat_roughness,
        inputs.sheen_color,
        inputs.sheen_roughness,
        inputs.anisotropy,
        inputs.iridescence,
        inputs.iridescence_thickness,
        inputs.transmission,
        inputs.thickness,
        inputs.specular_color,
        inputs.specular_intensity,
    ];
    if maps.iter().all(|map| map.is_none()) {
        return None;
    }

    let width = maps
        .iter()
        .flatten()
        .map(|map| map.width)
        .max()
        .unwrap_or(1);
    let height = maps
        .iter()
        .flatten()
        .map(|map| map.height)
        .max()
        .unwrap_or(1);
    let pixel_count = (width * height) as usize;
    let mut scalar = vec![255u8; pixel_count * 4];
    let mut sheen = vec![255u8; pixel_count * 4];
    let mut specular = vec![255u8; pixel_count * 4];
    let mut iridescence = vec![255u8; pixel_count * 4];
    // Default anisotropy map is direction +X, full strength.
    let mut anisotropy = vec![0u8; pixel_count * 4];
    for px in 0..pixel_count {
        anisotropy[px * 4] = 255;
        anisotropy[px * 4 + 1] = 128;
        anisotropy[px * 4 + 2] = 255;
        anisotropy[px * 4 + 3] = 255;
    }

    for y in 0..height {
        for x in 0..width {
            let out = ((y * width + x) * 4) as usize;
            if let Some(map) = inputs.clearcoat {
                scalar[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.clearcoat_is_srgb,
                );
            }
            if let Some(map) = inputs.clearcoat_roughness {
                scalar[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.clearcoat_roughness_is_srgb,
                );
            }
            if let Some(map) = inputs.transmission {
                scalar[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.transmission_is_srgb,
                );
            }
            if let Some(map) = inputs.thickness {
                scalar[out + 3] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.thickness_is_srgb,
                );
            }
            if let Some(map) = inputs.sheen_color {
                sheen[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.sheen_color_is_srgb,
                );
                sheen[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.sheen_color_is_srgb,
                );
                sheen[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.sheen_color_is_srgb,
                );
            }
            if let Some(map) = inputs.sheen_roughness {
                sheen[out + 3] = sample_texture_channel(map, x, y, width, height, 3);
            }
            if let Some(map) = inputs.anisotropy {
                anisotropy[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.anisotropy_is_srgb,
                );
                anisotropy[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.anisotropy_is_srgb,
                );
                anisotropy[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.anisotropy_is_srgb,
                );
            }
            if let Some(map) = inputs.iridescence {
                iridescence[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.iridescence_is_srgb,
                );
            }
            if let Some(map) = inputs.iridescence_thickness {
                iridescence[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.iridescence_thickness_is_srgb,
                );
            }
            if let Some(map) = inputs.specular_color {
                specular[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.specular_color_is_srgb,
                );
                specular[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.specular_color_is_srgb,
                );
                specular[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.specular_color_is_srgb,
                );
            }
            if let Some(map) = inputs.specular_intensity {
                specular[out + 3] = sample_texture_channel(map, x, y, width, height, 3);
            }
        }
    }

    let physical_layers_sampler = TextureSamplerSettings::first_from_textures(&[
        inputs.clearcoat,
        inputs.clearcoat_roughness,
        inputs.transmission,
        inputs.thickness,
        inputs.anisotropy,
        inputs.iridescence,
        inputs.iridescence_thickness,
    ]);
    let sheen_sampler =
        TextureSamplerSettings::first_from_textures(&[inputs.sheen_color, inputs.sheen_roughness]);
    let specular_sampler = TextureSamplerSettings::first_from_textures(&[
        inputs.specular_color,
        inputs.specular_intensity,
    ]);
    Some(PreparedPhysicalMaps {
        scalar_map: packed_texture(scalar, width, height, physical_layers_sampler),
        sheen_map: packed_texture(sheen, width, height, sheen_sampler),
        anisotropy_map: packed_texture(anisotropy, width, height, physical_layers_sampler),
        specular_map: packed_texture(specular, width, height, specular_sampler),
        iridescence_map: packed_texture(iridescence, width, height, physical_layers_sampler),
        physical_layers_sampler,
        sheen_sampler,
        specular_sampler,
    })
}

pub(super) fn packed_texture(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    sampler: TextureSamplerSettings,
) -> PreparedTexture {
    PreparedTexture {
        rgba,
        width,
        height,
        mipmaps: Vec::new(),
        wrap_s: sampler.wrap_s,
        wrap_t: sampler.wrap_t,
        mag_filter: sampler.mag_filter,
        min_filter: sampler.min_filter,
        mipmap_filter: sampler.mipmap_filter,
        anisotropy: sampler.anisotropy,
    }
}

pub(super) fn sample_texture_channel(
    texture: &PreparedTexture,
    x: u32,
    y: u32,
    out_width: u32,
    out_height: u32,
    channel: usize,
) -> u8 {
    let sx = (((x as f32 + 0.5) / out_width as f32) * texture.width as f32)
        .floor()
        .clamp(0.0, (texture.width - 1) as f32) as u32;
    let sy = (((y as f32 + 0.5) / out_height as f32) * texture.height as f32)
        .floor()
        .clamp(0.0, (texture.height - 1) as f32) as u32;
    texture.rgba[((sy * texture.width + sx) * 4) as usize + channel]
}

pub(super) fn sample_texture_color_channel(
    texture: &PreparedTexture,
    x: u32,
    y: u32,
    out_width: u32,
    out_height: u32,
    channel: usize,
    is_srgb: bool,
) -> u8 {
    let value = sample_texture_channel(texture, x, y, out_width, out_height, channel);
    if is_srgb && channel < 3 {
        srgb_u8_to_linear_u8(value)
    } else {
        value
    }
}

pub(super) fn srgb_u8_to_linear_u8(value: u8) -> u8 {
    (srgb_u8_to_linear_f32(value).clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(super) fn srgb_u8_to_linear_f32(value: u8) -> f32 {
    let channel = value as f32 / 255.0;
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

pub(super) fn decode_texture_rgb_srgb_to_linear(texture: &mut PreparedTexture) {
    for pixel in texture.rgba.chunks_exact_mut(4) {
        pixel[0] = srgb_u8_to_linear_u8(pixel[0]);
        pixel[1] = srgb_u8_to_linear_u8(pixel[1]);
        pixel[2] = srgb_u8_to_linear_u8(pixel[2]);
    }
    for mipmap in &mut texture.mipmaps {
        for pixel in mipmap.rgba.chunks_exact_mut(4) {
            pixel[0] = srgb_u8_to_linear_u8(pixel[0]);
            pixel[1] = srgb_u8_to_linear_u8(pixel[1]);
            pixel[2] = srgb_u8_to_linear_u8(pixel[2]);
        }
    }
}
