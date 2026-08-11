use super::*;

/// Resolve optional shadow casters from the scene into a shared depth array.
pub(super) fn resolve_shadow_maps(
    scene: &RenderScene,
    shadow_map_type: f32,
) -> Result<Option<ShadowMapSet>> {
    let Some(lights) = scene.lights.as_deref() else {
        return Ok(None);
    };
    let mut casters = Vec::new();
    let mut total_layers = 0u32;
    let mut atlas_width = 0u32;
    let mut atlas_height = 0u32;

    for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
        let light_type = light.light_type.to_ascii_lowercase();
        if light_type != "directional" && light_type != "spot" && light_type != "point" {
            continue;
        }
        if !light.cast_shadow.unwrap_or(false) {
            continue;
        }
        let prefix = format!("scene.lights[{i}]");

        let pos = parse_vec3(
            light.position.as_deref(),
            [0.0, 10.0, 0.0],
            &format!("{prefix}.position"),
        )?;
        let dir = parse_vec3(
            light.direction.as_deref(),
            [0.0, -1.0, 0.0],
            &format!("{prefix}.direction"),
        )?;
        let dir = if dir.length_squared() > 0.0 {
            dir.normalize()
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };

        let near = finite_f32(
            light.shadow_camera_near.unwrap_or(0.5),
            &format!("{prefix}.shadow.camera.near"),
        )?;
        let default_far = if light_type == "point" {
            let distance =
                finite_f32(light.distance.unwrap_or(0.0), &format!("{prefix}.distance"))?;
            if distance > 0.0 { distance } else { 500.0 }
        } else {
            500.0
        };
        let far = match light.shadow_camera_far {
            Some(value) => finite_f32(value, &format!("{prefix}.shadow.camera.far"))?,
            None => default_far,
        };
        if far <= near {
            bail!("{prefix}.shadow.camera has invalid near/far bounds");
        }
        let (map_width, map_height) = shadow_map_dimensions(light, &light_type, &prefix)?;

        let mut light_vps = [Mat4::IDENTITY; 6];
        let mut cascade_splits = [f32::MAX; 4];
        let mut layer_count = 1u32;
        let kind = if light_type == "point" {
            let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);
            let faces = [
                (Vec3::X, -Vec3::Y),
                (-Vec3::X, -Vec3::Y),
                (Vec3::Y, Vec3::Z),
                (-Vec3::Y, -Vec3::Z),
                (Vec3::Z, -Vec3::Y),
                (-Vec3::Z, -Vec3::Y),
            ];
            for (face, (face_dir, up)) in faces.into_iter().enumerate() {
                light_vps[face] = proj * Mat4::look_at_rh(pos, pos + face_dir, up);
            }
            layer_count = 6;
            ShadowKind::Point
        } else if light_type == "spot" {
            // View: look from light position along the light's direction. Pick an
            // up vector that is not collinear with `dir`.
            let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let view = Mat4::look_at_rh(pos, pos + dir, up);
            let angle = finite_f32(
                light.angle.unwrap_or(std::f64::consts::FRAC_PI_3),
                &format!("{prefix}.angle"),
            )?
            .clamp(0.001, std::f32::consts::FRAC_PI_2);
            let aspect = map_width as f32 / map_height as f32;
            let proj = Mat4::perspective_rh(
                (angle * 2.0).min(std::f32::consts::PI - 0.001),
                aspect,
                near,
                far,
            );
            light_vps[0] = proj * view;
            ShadowKind::DirectionalOrSpot
        } else {
            // View: look from light position along the light's direction. Pick an
            // up vector that is not collinear with `dir`.
            let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let view = Mat4::look_at_rh(pos, pos + dir, up);
            if let Some(bounds) = light.shadow_cascade_bounds.as_deref() {
                let cascade_count = (bounds.len() / 6).min(4);
                if cascade_count >= 2 {
                    for cascade in 0..cascade_count {
                        let base = cascade * 6;
                        let left = finite_f32(
                            bounds[base],
                            &format!("{prefix}.shadow.cascades[{cascade}].left"),
                        )?;
                        let right = finite_f32(
                            bounds[base + 1],
                            &format!("{prefix}.shadow.cascades[{cascade}].right"),
                        )?;
                        let top = finite_f32(
                            bounds[base + 2],
                            &format!("{prefix}.shadow.cascades[{cascade}].top"),
                        )?;
                        let bottom = finite_f32(
                            bounds[base + 3],
                            &format!("{prefix}.shadow.cascades[{cascade}].bottom"),
                        )?;
                        let cascade_near = finite_f32(
                            bounds[base + 4],
                            &format!("{prefix}.shadow.cascades[{cascade}].near"),
                        )?;
                        let cascade_far = finite_f32(
                            bounds[base + 5],
                            &format!("{prefix}.shadow.cascades[{cascade}].far"),
                        )?;
                        if right <= left || top <= bottom || cascade_far <= cascade_near {
                            bail!("{prefix}.shadow.cascades[{cascade}] has invalid bounds");
                        }
                        light_vps[cascade] = Mat4::orthographic_rh(
                            left,
                            right,
                            bottom,
                            top,
                            cascade_near,
                            cascade_far,
                        ) * view;
                    }
                    if let Some(splits) = light.shadow_cascade_splits.as_deref() {
                        for (slot, value) in splits.iter().take(cascade_count - 1).enumerate() {
                            cascade_splits[slot] = finite_f32(
                                *value,
                                &format!("{prefix}.shadow.cascadeSplits[{slot}]"),
                            )?
                            .max(0.0);
                        }
                    }
                    layer_count = cascade_count as u32;
                    ShadowKind::Cascaded
                } else {
                    // Orthographic bounds (three.js DirectionalLightShadow defaults: ±5).
                    let (left, right, top, bottom) = shadow_camera_bounds(light, &prefix)?;
                    if right <= left || top <= bottom {
                        bail!("{prefix}.shadow.camera has invalid orthographic bounds");
                    }
                    light_vps[0] =
                        Mat4::orthographic_rh(left, right, bottom, top, near, far) * view;
                    ShadowKind::DirectionalOrSpot
                }
            } else {
                // Orthographic bounds (three.js DirectionalLightShadow defaults: ±5).
                let (left, right, top, bottom) = shadow_camera_bounds(light, &prefix)?;
                if right <= left || top <= bottom {
                    bail!("{prefix}.shadow.camera has invalid orthographic bounds");
                }
                light_vps[0] = Mat4::orthographic_rh(left, right, bottom, top, near, far) * view;
                ShadowKind::DirectionalOrSpot
            }
        };

        let bias = finite_f32(
            light.shadow_bias.unwrap_or(0.0),
            &format!("{prefix}.shadow.bias"),
        )?;
        let normal_bias = finite_f32(
            light.shadow_normal_bias.unwrap_or(0.0),
            &format!("{prefix}.shadow.normalBias"),
        )?;
        let intensity = finite_f32(
            light.shadow_intensity.unwrap_or(1.0),
            &format!("{prefix}.shadow.intensity"),
        )?
        .max(0.0);
        let radius = shadow_radius(light, &prefix, shadow_map_type)?;

        let requested_layers = total_layers + layer_count;
        if requested_layers > MAX_SHADOW_LAYERS as u32 {
            bail!(
                "More than {MAX_SHADOW_LAYERS} shadow map layers are not supported by @headless-three/renderer yet ({requested_layers} requested). Directional and spot shadows use 1 layer, point shadows use 6 layers, and cascaded directional shadows use one layer per cascade."
            );
        }

        casters.push(ShadowCaster {
            light_vps,
            kind,
            light_index: i as u32,
            layer_base: total_layers,
            layer_count,
            cascade_splits,
            bias,
            normal_bias,
            radius,
            intensity,
        });
        total_layers = requested_layers;
        atlas_width = atlas_width.max(map_width);
        atlas_height = atlas_height.max(map_height);
    }
    if casters.is_empty() || !scene_has_shadow_caster_mesh(scene) {
        Ok(None)
    } else {
        Ok(Some(ShadowMapSet {
            casters,
            layer_count: total_layers,
            map_width: atlas_width,
            map_height: atlas_height,
        }))
    }
}

pub(super) fn scene_has_shadow_caster_mesh(scene: &RenderScene) -> bool {
    scene
        .meshes
        .as_deref()
        .is_some_and(|meshes| meshes.iter().any(|mesh| mesh.cast_shadow.unwrap_or(false)))
}

pub(super) fn shadow_radius(
    light: &crate::types::SceneLight,
    prefix: &str,
    shadow_map_type: f32,
) -> Result<f32> {
    let radius = finite_f32(
        light.shadow_radius.unwrap_or(1.0),
        &format!("{prefix}.shadow.radius"),
    )?
    .max(0.0);
    if (shadow_map_type - 3.0).abs() > f32::EPSILON {
        return Ok(radius);
    }

    let blur_samples = finite_f32(
        light.shadow_blur_samples.unwrap_or(8.0),
        &format!("{prefix}.shadow.blurSamples"),
    )?
    .max(0.0);
    if blur_samples <= 1.0 || radius <= 0.0 {
        return Ok(0.0);
    }
    Ok(radius * (blur_samples / 8.0).sqrt())
}

pub(super) fn shadow_camera_bounds(
    light: &crate::types::SceneLight,
    prefix: &str,
) -> Result<(f32, f32, f32, f32)> {
    Ok((
        finite_f32(
            light.shadow_camera_left.unwrap_or(-5.0),
            &format!("{prefix}.shadow.camera.left"),
        )?,
        finite_f32(
            light.shadow_camera_right.unwrap_or(5.0),
            &format!("{prefix}.shadow.camera.right"),
        )?,
        finite_f32(
            light.shadow_camera_top.unwrap_or(5.0),
            &format!("{prefix}.shadow.camera.top"),
        )?,
        finite_f32(
            light.shadow_camera_bottom.unwrap_or(-5.0),
            &format!("{prefix}.shadow.camera.bottom"),
        )?,
    ))
}

pub(super) fn shadow_map_dimensions(
    light: &crate::types::SceneLight,
    light_type: &str,
    prefix: &str,
) -> Result<(u32, u32)> {
    let width_hint = light.shadow_map_width.or(light.shadow_map_size);
    let height_hint = light.shadow_map_height.or(light.shadow_map_size);
    let width = width_hint.or(height_hint).unwrap_or(512).clamp(32, 4096);
    let height = height_hint.or(width_hint).unwrap_or(512).clamp(32, 4096);
    if light_type == "point" && width != height {
        bail!("{prefix}.shadow.mapSize must be square for point-light cube shadows");
    }
    Ok((width, height))
}
