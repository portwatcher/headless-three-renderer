use super::*;

pub fn prepare_meshes(scene: &crate::types::RenderScene) -> Result<Vec<PreparedMesh>> {
    if let Some(meshes) = scene.meshes.as_deref() {
        prepare_mesh_slice(meshes)
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn prepare_mesh_slice(meshes: &[SceneMesh]) -> Result<Vec<PreparedMesh>> {
    if meshes.len() < PARALLEL_MESH_PREPARE_THRESHOLD {
        return meshes.iter().enumerate().map(prepare_mesh).collect();
    }

    let worker_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(meshes.len());
    if worker_count <= 1 {
        return meshes.iter().enumerate().map(prepare_mesh).collect();
    }

    let chunk_size = meshes.len().div_ceil(worker_count);
    let mut chunks = thread::scope(|scope| {
        let handles = (0..worker_count)
            .filter_map(|worker_index| {
                let start = worker_index * chunk_size;
                if start >= meshes.len() {
                    return None;
                }
                let end = (start + chunk_size).min(meshes.len());
                Some(scope.spawn(move || {
                    let results = meshes[start..end]
                        .iter()
                        .enumerate()
                        .map(|(offset, mesh)| prepare_mesh((start + offset, mesh)))
                        .collect::<Vec<_>>();
                    (start, results)
                }))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("mesh preparation worker panicked"))
            .collect::<Vec<_>>()
    });
    chunks.sort_by_key(|(start, _)| *start);

    let mut prepared = Vec::with_capacity(meshes.len());
    for (_, results) in chunks {
        for result in results {
            prepared.push(result?);
        }
    }
    Ok(prepared)
}

pub fn texture_anisotropy(value: Option<f64>, field: &str) -> Result<u16> {
    match value {
        Some(value) => {
            let value = finite_f32(value, field)?;
            Ok(if value > 1.0 {
                value.floor().clamp(1.0, 16.0) as u16
            } else {
                1
            })
        }
        None => Ok(1),
    }
}

pub(super) fn prepare_mesh((mesh_index, mesh): (usize, &SceneMesh)) -> Result<PreparedMesh> {
    let topology = Topology::from_str_opt(mesh.topology.as_deref());
    let cached_native_mesh = mesh.positions.is_empty() && mesh.native_mesh_key.is_some();

    let min_positions = match topology {
        Topology::Triangles => 9, // at least 3 xyz
        Topology::Lines => 6,     // at least 2 xyz
        Topology::Points => 3,    // at least 1 xyz
    };
    if cached_native_mesh {
        if mesh.native_vertex_count.unwrap_or(0) == 0 {
            bail!(
                "scene.meshes[{mesh_index}].nativeVertexCount must be positive when using nativeMeshKey without positions"
            );
        }
    } else if mesh.positions.len() < min_positions || mesh.positions.len() % 3 != 0 {
        bail!(
            "scene.meshes[{mesh_index}].positions must contain at least {} xyz vertices",
            min_positions / 3
        );
    }

    let vertex_count = if cached_native_mesh {
        mesh.native_vertex_count.unwrap() as usize
    } else {
        mesh.positions.len() / 3
    };
    let material_color = parse_color(
        mesh.color.as_deref(),
        [0.82, 0.82, 0.82, 1.0],
        &format!("scene.meshes[{mesh_index}].color"),
    )?;

    let color_mode = if cached_native_mesh {
        ColorMode::new(None, vertex_count, material_color, mesh_index)?
    } else {
        ColorMode::new(
            mesh.colors.as_deref(),
            vertex_count,
            material_color,
            mesh_index,
        )?
    };

    let uvs = if cached_native_mesh {
        None
    } else {
        mesh.uvs.as_deref()
    };
    let has_uvs = uvs.map_or(false, |u| u.len() == vertex_count * 2);
    if let Some(u) = uvs {
        if u.len() != vertex_count * 2 {
            bail!(
                "scene.meshes[{mesh_index}].uvs has length {}; expected {}",
                u.len(),
                vertex_count * 2,
            );
        }
    }

    let uvs2 = if cached_native_mesh {
        None
    } else {
        mesh.uvs2.as_deref()
    };
    let has_uvs2 = uvs2.map_or(false, |u| u.len() == vertex_count * 2);
    if let Some(u) = uvs2 {
        if u.len() != vertex_count * 2 {
            bail!(
                "scene.meshes[{mesh_index}].uvs2 has length {}; expected {}",
                u.len(),
                vertex_count * 2,
            );
        }
    }

    let normals = if cached_native_mesh {
        None
    } else {
        mesh.normals.as_deref()
    };
    let has_normals = normals.map_or(false, |n| n.len() == vertex_count * 3);
    if let Some(n) = normals {
        if n.len() != vertex_count * 3 {
            bail!(
                "scene.meshes[{mesh_index}].normals has length {}; expected {}",
                n.len(),
                vertex_count * 3,
            );
        }
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    let normals_field = format!("scene.meshes[{mesh_index}].normals");
    let uvs_field = format!("scene.meshes[{mesh_index}].uvs");
    let uvs2_field = format!("scene.meshes[{mesh_index}].uvs2");
    if !cached_native_mesh {
        for vertex_index in 0..vertex_count {
            let base = vertex_index * 3;
            let uv_base = vertex_index * 2;
            vertices.push(Vertex {
                position: [
                    finite_f32(mesh.positions[base], "mesh position")?,
                    finite_f32(mesh.positions[base + 1], "mesh position")?,
                    finite_f32(mesh.positions[base + 2], "mesh position")?,
                ],
                normal: if has_normals {
                    let n = normals.unwrap();
                    [
                        finite_f32(n[base], &normals_field)?,
                        finite_f32(n[base + 1], &normals_field)?,
                        finite_f32(n[base + 2], &normals_field)?,
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                },
                tangent: [0.0, 0.0, 0.0, 0.0],
                color: color_mode.color(vertex_index),
                uv: if has_uvs {
                    let u = uvs.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs_field)?,
                        finite_f32(u[uv_base + 1], &uvs_field)?,
                    ]
                } else {
                    [0.0, 0.0]
                },
                uv2: if has_uvs2 {
                    let u = uvs2.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs2_field)?,
                        finite_f32(u[uv_base + 1], &uvs2_field)?,
                    ]
                } else if has_uvs {
                    let u = uvs.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs_field)?,
                        finite_f32(u[uv_base + 1], &uvs_field)?,
                    ]
                } else {
                    [0.0, 0.0]
                },
            });
        }
    }

    let mut indices = match &mesh.indices {
        Some(indices) => {
            let stride = match topology {
                Topology::Triangles => 3,
                Topology::Lines => 2,
                Topology::Points => 1,
            };
            if indices.len() < stride || indices.len() % stride != 0 {
                bail!(
                    "scene.meshes[{mesh_index}].indices must contain {} index/indices per primitive",
                    stride
                );
            }
            for &index in indices {
                if index as usize >= vertex_count {
                    bail!("scene.meshes[{mesh_index}].indices contains out-of-range index {index}");
                }
            }
            Some(indices.clone())
        }
        None if cached_native_mesh => None,
        None => {
            let stride = match topology {
                Topology::Triangles => 3,
                Topology::Lines => 2,
                Topology::Points => 1,
            };
            if vertex_count % stride != 0 {
                bail!(
                    "scene.meshes[{mesh_index}] has no indices, so positions must define complete primitives ({stride} vertices per primitive)"
                );
            }
            None
        }
    };

    let has_normal_map_slot = mesh
        .normal_map
        .as_ref()
        .is_some_and(|data| !data.is_empty());
    let use_flat_normals = mesh.flat_shading.unwrap_or(false)
        && !cached_native_mesh
        && !has_normal_map_slot
        && topology == Topology::Triangles;

    // If no normals were provided, compute vertex normals from triangle faces.
    // `flatShading` needs duplicated vertices for indexed geometry so each face
    // can carry its own normal. Lines and points stay unlit.
    if use_flat_normals {
        if let Some(flat_indices) = indices.take() {
            let mut expanded = Vec::with_capacity(flat_indices.len());
            for index in flat_indices {
                expanded.push(vertices[index as usize]);
            }
            vertices = expanded;
        }
        compute_flat_normals(&mut vertices, None);
    } else if !cached_native_mesh && !has_normals && topology == Topology::Triangles {
        compute_flat_normals(&mut vertices, indices.as_deref());
    }

    let (surface_texture_inputs, physical_texture_inputs) = thread::scope(|scope| -> Result<_> {
        let surface_handle = scope.spawn(|| prepare_surface_texture_inputs(mesh, mesh_index));
        let physical_handle = scope.spawn(|| prepare_physical_texture_inputs(mesh, mesh_index));

        let surface_texture_inputs = surface_handle
            .join()
            .expect("surface texture preparation worker panicked")?;
        let physical_texture_inputs = physical_handle
            .join()
            .expect("physical texture preparation worker panicked")?;

        Ok((surface_texture_inputs, physical_texture_inputs))
    })?;

    let SurfaceTextureInputs {
        texture,
        texture_transform,
        alpha_map_transform,
        normal_map_transform,
        bump_map_transform,
        matcap_map_transform,
        displacement_map_transform,
        displacement_map_is_srgb,
        metallic_roughness_texture_transform,
        metallic_roughness_texture_is_srgb,
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
        emissive_map_is_srgb,
        ao_map_is_srgb,
        light_map_is_srgb,
        specular_map_is_srgb,
        common,
    } = surface_texture_inputs;

    let CommonTextureInputs {
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
        alpha_map,
    } = common;

    let PhysicalTextureInputs {
        clearcoat_map,
        clearcoat_roughness_map,
        clearcoat_normal_map,
        sheen_color_map,
        sheen_roughness_map,
        anisotropy_map,
        iridescence_map,
        iridescence_thickness_map,
        transmission_map,
        thickness_map,
        specular_color_map,
        specular_intensity_map,
    } = physical_texture_inputs;

    let side = MeshSide::from_str_opt(mesh.side.as_deref());
    let normal_scale = if side == MeshSide::Back {
        [-normal_scale[0], -normal_scale[1]]
    } else {
        normal_scale
    };
    let bump_scale = if side == MeshSide::Back {
        -bump_scale
    } else {
        bump_scale
    };

    let physical_maps = pack_physical_maps(PhysicalMapInputs {
        clearcoat: clearcoat_map.as_ref(),
        clearcoat_roughness: clearcoat_roughness_map.as_ref(),
        sheen_color: sheen_color_map.as_ref(),
        sheen_roughness: sheen_roughness_map.as_ref(),
        anisotropy: anisotropy_map.as_ref(),
        iridescence: iridescence_map.as_ref(),
        iridescence_thickness: iridescence_thickness_map.as_ref(),
        transmission: transmission_map.as_ref(),
        thickness: thickness_map.as_ref(),
        specular_color: specular_color_map.as_ref(),
        specular_intensity: specular_intensity_map.as_ref(),
        clearcoat_is_srgb: matches!(mesh.clearcoat_map_color_space.as_deref(), Some("srgb")),
        clearcoat_roughness_is_srgb: matches!(
            mesh.clearcoat_roughness_map_color_space.as_deref(),
            Some("srgb")
        ),
        anisotropy_is_srgb: matches!(mesh.anisotropy_map_color_space.as_deref(), Some("srgb")),
        iridescence_is_srgb: matches!(mesh.iridescence_map_color_space.as_deref(), Some("srgb")),
        iridescence_thickness_is_srgb: matches!(
            mesh.iridescence_thickness_map_color_space.as_deref(),
            Some("srgb")
        ),
        transmission_is_srgb: matches!(mesh.transmission_map_color_space.as_deref(), Some("srgb")),
        thickness_is_srgb: matches!(mesh.thickness_map_color_space.as_deref(), Some("srgb")),
        sheen_color_is_srgb: matches!(mesh.sheen_color_map_color_space.as_deref(), Some("srgb")),
        specular_color_is_srgb: matches!(
            mesh.specular_color_map_color_space.as_deref(),
            Some("srgb")
        ),
    });

    let metallic = finite_clamp01(mesh.metallic.unwrap_or(0.0), "mesh metallic")?;
    let roughness = finite_clamp01(mesh.roughness.unwrap_or(1.0), "mesh roughness")?;
    let clearcoat = finite_clamp01(mesh.clearcoat.unwrap_or(0.0), "mesh clearcoat")?;
    let clearcoat_roughness = finite_clamp01(
        mesh.clearcoat_roughness.unwrap_or(0.0),
        "mesh clearcoatRoughness",
    )?;
    let clearcoat_normal_scale = parse_optional_vec2(
        mesh.clearcoat_normal_scale.as_deref(),
        [1.0, 1.0],
        "mesh clearcoatNormalScale",
    )?;
    let clearcoat_normal_scale = if side == MeshSide::Back {
        [-clearcoat_normal_scale[0], -clearcoat_normal_scale[1]]
    } else {
        clearcoat_normal_scale
    };
    let sheen_color = parse_optional_clamped_color3(
        mesh.sheen_color.as_deref(),
        [0.0, 0.0, 0.0],
        "mesh sheenColor",
    )?;
    let sheen_roughness =
        finite_clamp01(mesh.sheen_roughness.unwrap_or(1.0), "mesh sheenRoughness")?.max(0.0001);
    let anisotropy = finite_clamp01(mesh.anisotropy.unwrap_or(0.0), "mesh anisotropy")?;
    let anisotropy_rotation = finite_f32(
        mesh.anisotropy_rotation.unwrap_or(0.0),
        "mesh anisotropyRotation",
    )?;
    let iridescence = finite_clamp01(mesh.iridescence.unwrap_or(0.0), "mesh iridescence")?;
    let iridescence_ior =
        finite_f32(mesh.iridescence_ior.unwrap_or(1.3), "mesh iridescenceIor")?.clamp(1.0, 2.333);
    let iridescence_thickness_min = mesh
        .iridescence_thickness_min
        .map(|value| finite_f32(value, "mesh iridescenceThicknessMin"))
        .transpose()?
        .unwrap_or(100.0)
        .max(0.0);
    let iridescence_thickness_max = mesh
        .iridescence_thickness_max
        .map(|value| finite_f32(value, "mesh iridescenceThicknessMax"))
        .transpose()?
        .unwrap_or(400.0)
        .max(iridescence_thickness_min);
    let transmission = finite_clamp01(mesh.transmission.unwrap_or(0.0), "mesh transmission")?;
    let dispersion = finite_f32(mesh.dispersion.unwrap_or(0.0), "mesh dispersion")?.max(0.0);
    let ior = finite_f32(mesh.ior.unwrap_or(1.5), "mesh ior")?.clamp(1.0, 2.333);
    let thickness = finite_f32(mesh.thickness.unwrap_or(0.0), "mesh thickness")?.max(0.0);
    let attenuation_distance = mesh
        .attenuation_distance
        .map(|value| finite_f32(value, "mesh attenuationDistance"))
        .transpose()?
        .unwrap_or(1.0e20)
        .max(0.0)
        .min(1.0e20);
    let attenuation_color = parse_optional_clamped_color3(
        mesh.attenuation_color.as_deref(),
        [1.0, 1.0, 1.0],
        "mesh attenuationColor",
    )?;
    let physical_specular_color = parse_optional_clamped_color3(
        mesh.physical_specular_color.as_deref(),
        [1.0, 1.0, 1.0],
        "mesh specularColor",
    )?;
    let physical_specular_intensity = finite_clamp01(
        mesh.physical_specular_intensity.unwrap_or(1.0),
        "mesh specularIntensity",
    )?;
    let distance_reference_position = parse_optional_vec3(
        mesh.distance_reference_position.as_deref(),
        &format!("scene.meshes[{mesh_index}].distanceReferencePosition"),
    )?;
    let distance_near = mesh
        .distance_near
        .map(|value| finite_f32(value, "mesh distanceNear"))
        .transpose()?;
    let distance_far = mesh
        .distance_far
        .map(|value| finite_f32(value, "mesh distanceFar"))
        .transpose()?;
    let specular_color = parse_optional_clamped_color3(
        mesh.specular_color.as_deref(),
        [17.0 / 255.0, 17.0 / 255.0, 17.0 / 255.0],
        "mesh specular",
    )?;
    let shininess = finite_f32(mesh.shininess.unwrap_or(30.0), "mesh shininess")?.max(0.0001);

    if !cached_native_mesh && let Some(displacement_map) = displacement_map.as_ref() {
        if has_uvs && topology == Topology::Triangles {
            apply_displacement_map(
                &mut vertices,
                displacement_map,
                displacement_map_transform,
                displacement_map_is_srgb,
                mesh.displacement_map_uses_uv2.unwrap_or(false),
                displacement_scale,
                displacement_bias,
            );
        }
    }

    // Compute tangents when normal/bump mapping or anisotropic shading needs a frame.
    if !cached_native_mesh
        && (normal_map.is_some()
            || bump_map.is_some()
            || clearcoat_normal_map.is_some()
            || anisotropy > 0.0)
        && has_uvs
        && topology == Topology::Triangles
    {
        compute_tangents(&mut vertices, mesh.indices.as_deref());
    }

    let emissive_intensity = finite_f32(
        mesh.emissive_intensity.unwrap_or(1.0),
        "mesh emissiveIntensity",
    )?;
    let emissive_color =
        parse_optional_clamped_color3(mesh.emissive.as_deref(), [0.0, 0.0, 0.0], "mesh emissive")?;
    let emissive = [
        emissive_color[0] * emissive_intensity,
        emissive_color[1] * emissive_intensity,
        emissive_color[2] * emissive_intensity,
    ];

    let alpha_test = finite_clamp01(mesh.alpha_test.unwrap_or(0.0), "mesh alphaTest")?;
    let alpha_hash = mesh.alpha_hash.unwrap_or(false);
    let alpha_to_coverage = mesh.alpha_to_coverage.unwrap_or(false);
    let premultiplied_alpha = mesh.premultiplied_alpha.unwrap_or(false);
    let tone_mapped = mesh.tone_mapped.unwrap_or(true);
    let fog = mesh.fog.unwrap_or(true);
    let (clipping_planes, clipping_plane_count, clipping_union_count) = parse_clipping_planes(
        mesh.clipping_planes.as_deref(),
        mesh.clipping_union_count,
        mesh_index,
    )?;
    let is_transparent = (!alpha_hash && mesh.transparent.unwrap_or(material_color[3] < 0.999))
        || transmission > 0.0001;
    let blending = BlendMode::from_str_opt(mesh.blending.as_deref());
    let custom_blend = parse_custom_blend_state(mesh, blending, mesh_index)?;
    let depth_test = mesh.depth_test.unwrap_or(true);
    let depth_func = parse_depth_func(mesh.depth_func.as_deref(), mesh_index)?;
    let depth_write = depth_test && mesh.depth_write.unwrap_or(true);
    let color_write = mesh.color_write.unwrap_or(true);
    let polygon_offset = mesh.polygon_offset.unwrap_or(false);
    let polygon_offset_factor = finite_f32(
        mesh.polygon_offset_factor.unwrap_or(0.0),
        "mesh polygonOffsetFactor",
    )?;
    let polygon_offset_units = finite_f32(
        mesh.polygon_offset_units.unwrap_or(0.0),
        "mesh polygonOffsetUnits",
    )?
    .round()
    .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    let stencil_write = mesh.stencil_write.unwrap_or(false);
    let stencil_ref = mesh.stencil_ref.unwrap_or(0).min(0xff);
    let stencil_func_mask = mesh.stencil_func_mask.unwrap_or(0xff).min(0xff);
    let stencil_write_mask = mesh.stencil_write_mask.unwrap_or(0xff).min(0xff);
    let stencil_func =
        parse_stencil_compare(mesh.stencil_func.unwrap_or(519), mesh_index, "stencilFunc")?;
    let stencil_fail =
        parse_stencil_operation(mesh.stencil_fail.unwrap_or(7680), mesh_index, "stencilFail")?;
    let stencil_z_fail = parse_stencil_operation(
        mesh.stencil_z_fail.unwrap_or(7680),
        mesh_index,
        "stencilZFail",
    )?;
    let stencil_z_pass = parse_stencil_operation(
        mesh.stencil_z_pass.unwrap_or(7680),
        mesh_index,
        "stencilZPass",
    )?;
    let shadow_side = parse_shadow_side(mesh.shadow_side.as_deref(), mesh_index)?;
    let requested_shading_model = ShadingModel::from_str_opt(mesh.shading_model.as_deref());
    // Lines and points are normally unlit. Depth/distance wireframe meshes are
    // emitted as lines but still use position-only material outputs.
    let shading_model = match topology {
        Topology::Triangles => requested_shading_model,
        Topology::Lines
            if matches!(
                requested_shading_model,
                ShadingModel::Depth | ShadingModel::Distance
            ) =>
        {
            requested_shading_model
        }
        Topology::Lines | Topology::Points => ShadingModel::Basic,
    };
    let environment_map_intensity = match mesh.environment_map_intensity {
        Some(value) => Some(finite_f32(value, "mesh environmentMapIntensity")?),
        None => None,
    };
    let environment_map_combine = mesh.environment_map_combine.unwrap_or(0);
    if environment_map_combine > 2 {
        bail!("mesh {mesh_index} environmentMapCombine must be 0, 1, or 2");
    }
    let environment_map_reflectivity = match mesh.environment_map_reflectivity {
        Some(value) => finite_f32(value, "mesh environmentMapReflectivity")?,
        None => 1.0,
    };
    let environment_map_refraction = mesh.environment_map_refraction.unwrap_or(false);
    let environment_map_refraction_ratio = match mesh.environment_map_refraction_ratio {
        Some(value) => finite_f32(value, "mesh environmentMapRefractionRatio")?,
        None => 0.98,
    };

    Ok(PreparedMesh {
        native_mesh_key: mesh.native_mesh_key,
        vertices,
        indices,
        transform: parse_transform(mesh.transform.as_deref(), mesh_index)?,
        texture,
        normal_map,
        normal_map_type,
        normal_scale,
        bump_map,
        bump_scale,
        matcap_map,
        gradient_map,
        metallic_roughness_texture,
        specular_map,
        emissive_map,
        ao_map,
        light_map,
        alpha_map,
        physical_maps,
        clearcoat_normal_map,
        ao_map_intensity,
        light_map_intensity,
        metallic,
        roughness,
        clearcoat,
        clearcoat_roughness,
        clearcoat_normal_scale,
        sheen_color,
        sheen_roughness,
        anisotropy,
        anisotropy_rotation,
        iridescence,
        iridescence_ior,
        iridescence_thickness_min,
        iridescence_thickness_max,
        transmission,
        dispersion,
        ior,
        thickness,
        attenuation_distance,
        attenuation_color,
        physical_specular_color,
        physical_specular_intensity,
        distance_reference_position,
        distance_near,
        distance_far,
        specular_color,
        shininess,
        emissive,
        base_color: color_to_f32(material_color),
        depth_packing: DepthPacking::from_three_constant(mesh.depth_packing),
        alpha_test,
        alpha_hash,
        alpha_to_coverage,
        premultiplied_alpha,
        tone_mapped,
        fog,
        clipping_planes,
        clipping_plane_count,
        clipping_union_count,
        clip_shadows: mesh.clip_shadows.unwrap_or(false),
        is_transparent,
        blending,
        custom_blend,
        depth_test,
        depth_func,
        depth_write,
        color_write,
        polygon_offset,
        polygon_offset_factor,
        polygon_offset_units,
        stencil_write,
        stencil_ref,
        stencil_func_mask,
        stencil_write_mask,
        stencil_func,
        stencil_fail,
        stencil_z_fail,
        stencil_z_pass,
        side,
        shadow_side,
        shading_model,
        use_environment_map: mesh.use_environment_map,
        environment_map_intensity,
        environment_map_combine,
        environment_map_reflectivity,
        environment_map_refraction,
        environment_map_refraction_ratio,
        topology,
        custom_fragment_shader: mesh
            .custom_fragment_shader
            .as_deref()
            .map(str::trim)
            .filter(|source| !source.is_empty())
            .map(str::to_owned),
        texture_transform,
        texture_uses_uv2: mesh.texture_uses_uv2.unwrap_or(false),
        alpha_map_transform,
        alpha_map_is_srgb,
        alpha_map_uses_uv2: mesh.alpha_map_uses_uv2.unwrap_or(false),
        normal_map_transform,
        normal_map_uses_uv2: mesh.normal_map_uses_uv2.unwrap_or(false),
        bump_map_transform,
        bump_map_uses_uv2: mesh.bump_map_uses_uv2.unwrap_or(false),
        matcap_map_transform,
        matcap_map_uses_uv2: mesh.matcap_map_uses_uv2.unwrap_or(false),
        metallic_roughness_texture_transform,
        metallic_roughness_texture_is_srgb,
        metallic_roughness_texture_uses_uv2: mesh
            .metallic_roughness_texture_uses_uv2
            .unwrap_or(false),
        emissive_map_transform,
        emissive_map_uses_uv2: mesh.emissive_map_uses_uv2.unwrap_or(false),
        ao_map_transform,
        ao_map_is_srgb,
        ao_map_uses_uv2: mesh.ao_map_uses_uv2.unwrap_or(false),
        light_map_transform,
        light_map_uses_uv2: mesh.light_map_uses_uv2.unwrap_or(false),
        specular_map_transform,
        specular_map_is_srgb,
        specular_map_uses_uv2: mesh.specular_map_uses_uv2.unwrap_or(false),
        specular_color_map_transform,
        specular_color_map_uses_uv2: mesh.specular_color_map_uses_uv2.unwrap_or(false),
        specular_intensity_map_transform,
        specular_intensity_map_uses_uv2: mesh.specular_intensity_map_uses_uv2.unwrap_or(false),
        clearcoat_map_transform,
        clearcoat_map_uses_uv2: mesh.clearcoat_map_uses_uv2.unwrap_or(false),
        clearcoat_roughness_map_transform,
        clearcoat_roughness_map_uses_uv2: mesh.clearcoat_roughness_map_uses_uv2.unwrap_or(false),
        clearcoat_normal_map_transform,
        clearcoat_normal_map_uses_uv2: mesh.clearcoat_normal_map_uses_uv2.unwrap_or(false),
        sheen_color_map_transform,
        sheen_color_map_uses_uv2: mesh.sheen_color_map_uses_uv2.unwrap_or(false),
        sheen_roughness_map_transform,
        sheen_roughness_map_uses_uv2: mesh.sheen_roughness_map_uses_uv2.unwrap_or(false),
        anisotropy_map_transform,
        anisotropy_map_uses_uv2: mesh.anisotropy_map_uses_uv2.unwrap_or(false),
        iridescence_map_transform,
        iridescence_map_uses_uv2: mesh.iridescence_map_uses_uv2.unwrap_or(false),
        iridescence_thickness_map_transform,
        iridescence_thickness_map_uses_uv2: mesh
            .iridescence_thickness_map_uses_uv2
            .unwrap_or(false),
        transmission_map_transform,
        transmission_map_uses_uv2: mesh.transmission_map_uses_uv2.unwrap_or(false),
        thickness_map_transform,
        thickness_map_uses_uv2: mesh.thickness_map_uses_uv2.unwrap_or(false),
        texture_is_srgb,
        matcap_map_is_srgb,
        gradient_map_is_srgb,
        emissive_map_is_srgb,
        light_map_is_srgb,
        cast_shadow: mesh.cast_shadow.unwrap_or(false),
        receive_shadow: mesh.receive_shadow.unwrap_or(false),
        group_order: finite_f32(mesh.group_order.unwrap_or(0.0), "mesh groupOrder")?,
        render_order: finite_f32(mesh.render_order.unwrap_or(0.0), "mesh renderOrder")?,
        sort_z: finite_f32(mesh.sort_z.unwrap_or(0.0), "mesh sortZ")?,
        sort_index: mesh.sort_index.unwrap_or(mesh_index as u32),
        material_variant: mesh.material_variant.unwrap_or(0),
        material_sort_key: mesh.material_sort_key.unwrap_or(0),
    })
}
