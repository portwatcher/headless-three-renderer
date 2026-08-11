use super::*;

impl RenderSettings {
    pub fn from_scene(scene: &RenderScene, camera: &Camera, limits: wgpu::Limits) -> Result<Self> {
        let width = camera.width.or(scene.width).unwrap_or(DEFAULT_WIDTH);
        let height = camera.height.or(scene.height).unwrap_or(DEFAULT_HEIGHT);

        validate_dimension("width", width, limits.max_texture_dimension_2d)?;
        validate_dimension("height", height, limits.max_texture_dimension_2d)?;

        let viewport = parse_viewport_rect(scene.viewport.as_deref(), width, height)?;
        let scissor = parse_scissor_rect(scene.scissor.as_deref(), width, height)?;

        let background = parse_color(
            scene.background.as_deref(),
            [0.04, 0.045, 0.05, 1.0],
            "scene.background",
        )?;
        let background_intensity = finite_f32(
            scene.background_intensity.unwrap_or(1.0),
            "scene.backgroundIntensity",
        )?
        .max(0.0);
        thread::scope(|scope| {
            let background_handle =
                scope.spawn(|| prepare_background_texture(scene, background_intensity));
            let ibl_handle = scope.spawn(|| prepare_scene_ibl(scene));

            let background_texture =
                join_prepare_worker(background_handle, "background texture worker")?;

            let eye = parse_vec3(camera.eye.as_deref(), [2.5, 1.8, 3.2], "camera.eye")?;
            let target = parse_vec3(camera.target.as_deref(), [0.0, 0.0, 0.0], "camera.target")?;
            let up = parse_vec3(camera.up.as_deref(), [0.0, 1.0, 0.0], "camera.up")?;

            if eye.distance_squared(target) <= f32::EPSILON {
                bail!("camera.eye must not equal camera.target");
            }
            if up.length_squared() <= f32::EPSILON {
                bail!("camera.up must not be a zero vector");
            }

            let fov_y_degrees = camera.fov_y_degrees.unwrap_or(45.0);
            if !fov_y_degrees.is_finite() || !(1.0..179.0).contains(&fov_y_degrees) {
                bail!("camera.fov_y_degrees must be finite and between 1 and 179");
            }

            let near = finite_f32(camera.near.unwrap_or(0.01), "camera.near")?;
            if near < 0.0 {
                bail!("camera.near must be greater than or equal to 0");
            }
            let far = finite_positive(camera.far.unwrap_or(100.0), "camera.far")?;
            if far <= near {
                bail!("camera.far must be greater than camera.near");
            }

            let view = match camera.view_matrix.as_deref() {
                Some(matrix) => parse_mat4(matrix, "camera.viewMatrix")?,
                None => Mat4::look_at_rh(eye, target, up.normalize()),
            };

            let view_projection = match camera.view_projection.as_deref() {
                Some(matrix) => parse_mat4(matrix, "camera.viewProjection")?,
                None => {
                    let aspect = width as f32 / height as f32;
                    let projection =
                        Mat4::perspective_rh(fov_y_degrees.to_radians() as f32, aspect, near, far);
                    projection * view
                }
            };

            let camera_pos = parse_vec3(
                camera.camera_position.as_deref(),
                eye.to_array(),
                "camera.cameraPosition",
            )?;

            let output_format = OutputFormat::from_scene(scene)?;
            let output_color_space = OutputColorSpace::from_scene(scene)?;
            let tone_mapping = resolve_tone_mapping(scene.tone_mapping)?;
            let tone_mapping_exposure = finite_f32(
                scene.tone_mapping_exposure.unwrap_or(1.0),
                "scene.toneMappingExposure",
            )?;
            if tone_mapping_exposure < 0.0 {
                bail!("scene.toneMappingExposure must be non-negative");
            }
            let transmission_resolution_scale = finite_positive(
                scene.transmission_resolution_scale.unwrap_or(1.0),
                "scene.transmissionResolutionScale",
            )?;
            validate_transmission_resolution_scale(
                width,
                height,
                transmission_resolution_scale,
                limits.max_texture_dimension_2d,
            )?;
            let sample_count = resolve_sample_count(scene.sample_count)?;
            let shadow_map_type = resolve_shadow_map_type(scene.shadow_map_type)?;
            let lights = prepare_lights(scene)?;
            let ambient_color = parse_color(
                scene.ambient_light.as_deref(),
                [1.0, 1.0, 1.0, 1.0],
                "scene.ambientLight",
            )?;
            let ambient_intensity = finite_f32(
                scene.ambient_intensity.unwrap_or(0.0),
                "scene.ambientIntensity",
            )?;
            let (light_probe, has_light_probe) = parse_light_probe(scene.light_probe.as_deref())?;
            let ibl = join_prepare_worker(ibl_handle, "environment map worker")?;
            let env_intensity = finite_f32(
                scene.environment_map_intensity.unwrap_or(1.0),
                "scene.environmentIntensity",
            )?;

            let fog = FogSettings::from_scene(scene, background)?;
            let shadow = resolve_shadow_maps(scene, shadow_map_type)?;
            let post_processing = PostProcessingSettings::from_scene(scene)?;

            Ok(Self {
                width,
                height,
                background,
                background_intensity,
                viewport,
                scissor,
                background_texture,
                output_format,
                output_color_space,
                tone_mapping,
                tone_mapping_exposure,
                transmission_resolution_scale,
                sample_count,
                shadow_map_type,
                view,
                view_projection,
                camera_pos,
                near,
                far,
                lights,
                ambient_color: [
                    ambient_color[0] as f32,
                    ambient_color[1] as f32,
                    ambient_color[2] as f32,
                ],
                ambient_intensity,
                light_probe,
                has_light_probe,
                ibl,
                env_intensity,
                fog,
                shadow,
                post_processing,
            })
        })
    }
}

pub(super) fn join_prepare_worker<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T>>,
    label: &str,
) -> Result<T> {
    handle.join().unwrap_or_else(|_| panic!("{label} panicked"))
}

pub(super) fn prepare_background_texture(
    scene: &RenderScene,
    background_intensity: f32,
) -> Result<Option<BackgroundTexture>> {
    match &scene.background_texture {
        Some(data) if !data.is_empty() => {
            let mut texture = decode_texture_with_label(
                data,
                scene.background_texture_width,
                scene.background_texture_height,
                "scene.backgroundTexture",
            )?;
            texture.wrap_s = WrapMode::from_str_opt(scene.background_texture_wrap_s.as_deref());
            texture.wrap_t = WrapMode::from_str_opt(scene.background_texture_wrap_t.as_deref());
            texture.mag_filter =
                TextureFilter::from_str_opt(scene.background_texture_mag_filter.as_deref());
            texture.min_filter =
                TextureFilter::from_min_filter_str(scene.background_texture_min_filter.as_deref());
            texture.mipmap_filter =
                MipmapFilter::from_min_filter_str(scene.background_texture_min_filter.as_deref());
            texture.anisotropy = texture_anisotropy(
                scene.background_texture_anisotropy,
                "scene.backgroundTextureAnisotropy",
            )?;
            Ok(Some(BackgroundTexture {
                texture,
                transform: parse_texture_transform(
                    scene.background_texture_transform.as_deref(),
                    "scene.backgroundTextureTransform",
                )?,
                is_srgb: matches!(
                    scene.background_texture_color_space.as_deref(),
                    Some("srgb")
                ),
                mapping: BackgroundTextureMapping::from_scene(
                    scene.background_texture_mapping.as_deref(),
                )?,
                rotation: parse_rotation_columns(
                    scene.background_texture_rotation.as_deref(),
                    "scene.backgroundTextureRotation",
                )?,
                intensity: background_intensity,
                blurriness: finite_f32(
                    scene.background_texture_blurriness.unwrap_or(0.0),
                    "scene.backgroundTextureBlurriness",
                )?
                .clamp(0.0, 1.0),
            }))
        }
        _ => Ok(None),
    }
}

pub(super) fn validate_transmission_resolution_scale(
    width: u32,
    height: u32,
    scale: f32,
    max_texture_dimension: u32,
) -> Result<()> {
    let scaled_width = (width as f64 * f64::from(scale)).round();
    let scaled_height = (height as f64 * f64::from(scale)).round();
    if scaled_width < 1.0 || scaled_height < 1.0 {
        bail!("scene.transmissionResolutionScale must produce at least a 1x1 texture");
    }
    if scaled_width > f64::from(max_texture_dimension)
        || scaled_height > f64::from(max_texture_dimension)
    {
        bail!(
            "scene.transmissionResolutionScale produces an intermediate texture above max_texture_dimension_2d {max_texture_dimension}"
        );
    }
    Ok(())
}

pub(super) fn prepare_scene_ibl(scene: &RenderScene) -> Result<Option<IblMaps>> {
    match &scene.environment_map {
        Some(data) if !data.is_empty() => {
            let rotation = parse_rotation_columns(
                scene.environment_map_rotation.as_deref(),
                "scene.environmentMapRotation",
            )?;
            let env_map = EnvMap::from_bytes(
                data,
                scene.environment_map_width,
                scene.environment_map_height,
                parse_environment_color_space(scene.environment_map_color_space.as_deref())?,
            )?;
            Ok(Some(compute_ibl(&env_map, rotation)))
        }
        _ => Ok(None),
    }
}

pub(super) fn parse_viewport_rect(
    values: Option<&[f64]>,
    target_width: u32,
    target_height: u32,
) -> Result<Option<ViewportRect>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 4 {
        bail!("scene.viewport must be `[x, y, width, height]`");
    }

    let x = finite_f32(values[0], "scene.viewport")?;
    let y = finite_f32(values[1], "scene.viewport")?;
    let width = finite_f32(values[2], "scene.viewport")?;
    let height = finite_f32(values[3], "scene.viewport")?;
    if x < 0.0 || y < 0.0 {
        bail!("scene.viewport x and y must be greater than or equal to 0");
    }
    if width <= 0.0 || height <= 0.0 {
        bail!("scene.viewport width and height must be greater than 0");
    }
    if x + width > target_width as f32 || y + height > target_height as f32 {
        bail!("scene.viewport must fit inside the render target");
    }

    Ok(Some(ViewportRect {
        x,
        y,
        width,
        height,
    }))
}

pub(super) fn parse_scissor_rect(
    values: Option<&[f64]>,
    target_width: u32,
    target_height: u32,
) -> Result<Option<ScissorRect>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 4 {
        bail!("scene.scissor must be `[x, y, width, height]`");
    }

    let x = finite_u32_pixel(values[0], "scene.scissor")?;
    let y = finite_u32_pixel(values[1], "scene.scissor")?;
    let width = finite_u32_pixel(values[2], "scene.scissor")?;
    let height = finite_u32_pixel(values[3], "scene.scissor")?;
    if width == 0 || height == 0 {
        bail!("scene.scissor width and height must be greater than 0");
    }
    if x.checked_add(width)
        .is_none_or(|right| right > target_width)
        || y.checked_add(height)
            .is_none_or(|bottom| bottom > target_height)
    {
        bail!("scene.scissor must fit inside the render target");
    }

    Ok(Some(ScissorRect {
        x,
        y,
        width,
        height,
    }))
}

pub(super) fn finite_u32_pixel(value: f64, field: &str) -> Result<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 || value.fract() != 0.0 {
        bail!("{field} must contain non-negative whole pixel numbers");
    }
    Ok(value as u32)
}
