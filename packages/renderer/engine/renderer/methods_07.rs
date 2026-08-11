use super::*;

impl GpuRenderer {
    pub(super) fn upload_mesh(
        &self,
        settings: &RenderSettings,
        mesh: &PreparedMesh,
        uniform_slot: usize,
        dynamic_uniform_slots: &mut Vec<CachedDynamicUniformBindGroup>,
    ) -> Result<GpuMesh> {
        let CachedMeshBuffers {
            vertex_buffer,
            index_buffer,
            vertex_count,
            index_count,
        } = self.mesh_buffers_for(mesh)?;

        let model = mesh.transform;
        let mvp = settings.view_projection * model;
        let normal_matrix = model.inverse().transpose();

        let mut lights = [GpuLight::zeroed(); MAX_LIGHTS];
        for (i, light) in settings.lights.iter().take(MAX_LIGHTS).enumerate() {
            lights[i] = *light;
        }

        let physical_params2 = if mesh.shading_model == ShadingModel::Phong {
            [
                mesh.specular_color[0],
                mesh.specular_color[1],
                mesh.specular_color[2],
                mesh.shininess,
            ]
        } else {
            [
                mesh.sheen_color[0],
                mesh.sheen_color[1],
                mesh.sheen_color[2],
                mesh.sheen_roughness,
            ]
        };
        let distance_reference_position = mesh.distance_reference_position.unwrap_or([
            settings.camera_pos.x,
            settings.camera_pos.y,
            settings.camera_pos.z,
        ]);
        let distance_near = mesh.distance_near.unwrap_or(settings.near);
        let distance_far = mesh.distance_far.unwrap_or(settings.far);
        let mesh_has_ibl = settings.ibl.is_some() && mesh.use_environment_map.unwrap_or(true);
        let mesh_env_intensity = mesh
            .environment_map_intensity
            .unwrap_or(settings.env_intensity);

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            view: settings.view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
            camera_pos: [
                settings.camera_pos.x,
                settings.camera_pos.y,
                settings.camera_pos.z,
                0.0,
            ],
            base_color: mesh.base_color,
            emissive: [
                mesh.emissive[0],
                mesh.emissive[1],
                mesh.emissive[2],
                mesh.alpha_test,
            ],
            metallic: mesh.metallic,
            roughness: mesh.roughness,
            ambient_intensity: settings.ambient_intensity,
            num_lights: settings.lights.len().min(MAX_LIGHTS) as u32,
            ambient_color: [
                settings.ambient_color[0],
                settings.ambient_color[1],
                settings.ambient_color[2],
                0.0,
            ],
            light_probe: light_probe_rows(settings),
            light_probe_params: [
                if settings.has_light_probe { 1.0 } else { 0.0 },
                if mesh.gradient_map.is_some() {
                    1.0
                } else {
                    0.0
                },
                mesh.depth_packing.as_u32() as f32,
                if mesh.matcap_map.is_some() { 1.0 } else { 0.0 },
            ],
            normal_map_params: [
                if mesh.normal_map.is_some() {
                    mesh.normal_scale[0]
                } else {
                    mesh.bump_scale
                },
                if mesh.normal_map.is_some() {
                    mesh.normal_scale[1]
                } else {
                    0.0
                },
                if mesh.normal_map.is_some() {
                    mesh.normal_map_type.normal_mode()
                } else if mesh.bump_map.is_some() {
                    2.0
                } else {
                    0.0
                },
                if mesh_has_ibl { 1.0 } else { 0.0 },
            ],
            ibl_params: [
                mesh_env_intensity,
                mesh.shading_model.as_u32() as f32,
                settings.near,
                settings.far,
            ],
            env_map_params: [
                mesh.environment_map_combine as f32,
                mesh.environment_map_reflectivity,
                if mesh.use_environment_map == Some(true)
                    && matches!(
                        mesh.shading_model,
                        ShadingModel::Basic | ShadingModel::Lambert | ShadingModel::Phong
                    )
                {
                    if mesh.environment_map_refraction {
                        2.0
                    } else {
                        1.0
                    }
                } else {
                    0.0
                },
                mesh.environment_map_refraction_ratio,
            ],
            ao_params: [
                mesh.ao_map_intensity,
                if mesh.ao_map.is_some() { 1.0 } else { 0.0 },
                if mesh.alpha_map.is_some() { 1.0 } else { 0.0 },
                if mesh.light_map.is_some() { 1.0 } else { 0.0 },
            ],
            render_params: [
                1.0 / settings.width as f32,
                1.0 / settings.height as f32,
                settings.width as f32,
                settings.height as f32,
            ],
            output_params: [
                if settings.output_color_space.is_linear() {
                    1.0
                } else {
                    0.0
                },
                if mesh.tone_mapped {
                    settings.tone_mapping
                } else {
                    0.0
                },
                if mesh.alpha_to_coverage && settings.sample_count > 1 {
                    1.0
                } else {
                    0.0
                },
                settings.tone_mapping_exposure,
            ],
            texture_transform1: [
                mesh.texture_transform[0],
                mesh.texture_transform[1],
                mesh.texture_transform[2],
                if mesh.texture_uses_uv2 { 1.0 } else { 0.0 },
            ],
            texture_transform2: [
                mesh.texture_transform[3],
                mesh.texture_transform[4],
                mesh.texture_transform[5],
                if mesh.texture_is_srgb { 1.0 } else { 0.0 },
            ],
            alpha_map_transform1: [
                mesh.alpha_map_transform[0],
                mesh.alpha_map_transform[1],
                mesh.alpha_map_transform[2],
                if mesh.alpha_map_is_srgb { 1.0 } else { 0.0 },
            ],
            alpha_map_transform2: [
                mesh.alpha_map_transform[3],
                mesh.alpha_map_transform[4],
                mesh.alpha_map_transform[5],
                if mesh.alpha_map_uses_uv2 { 1.0 } else { 0.0 },
            ],
            map_transform_rows: map_transform_rows(mesh),
            physical_map_transform_rows: physical_map_transform_rows(mesh),
            clipping_planes: mesh.clipping_planes,
            clipping_params: [
                mesh.clipping_union_count as f32,
                mesh.clipping_plane_count as f32,
                if mesh.alpha_hash { 1.0 } else { 0.0 },
                if mesh.premultiplied_alpha { 1.0 } else { 0.0 },
            ],
            fog_color: [
                settings.fog.color[0],
                settings.fog.color[1],
                settings.fog.color[2],
                0.0,
            ],
            fog_params: [
                if settings.fog.active && mesh.fog {
                    settings.fog.mode
                } else {
                    0.0
                },
                settings.fog.near,
                settings.fog.far,
                settings.fog.density,
            ],
            light_space_matrices: shadow_light_space_matrices(settings),
            shadow_params: shadow_params(settings, mesh.receive_shadow),
            shadow_params2: shadow_params2(settings),
            shadow_params3: shadow_params3(settings),
            shadow_params4: settings
                .shadow
                .as_ref()
                .map(|s| {
                    let radius = s.casters.first().map_or(1.0, |caster| caster.radius);
                    [
                        radius,
                        if mesh.clip_shadows { 1.0 } else { 0.0 },
                        shadow_side_mode(mesh.shadow_side),
                        shadow_alpha_cutoff(mesh),
                    ]
                })
                .unwrap_or([1.0, 0.0, 0.0, 0.0]),
            shadow_infos: shadow_infos(settings),
            shadow_biases: shadow_biases(settings),
            shadow_cascade_splits: shadow_cascade_splits(settings),
            physical_params1: [
                mesh.clearcoat,
                mesh.clearcoat_roughness,
                mesh.transmission,
                mesh.ior,
            ],
            physical_params2,
            physical_params3: if mesh.shading_model == ShadingModel::Distance {
                [
                    mesh.anisotropy,
                    mesh.anisotropy_rotation,
                    distance_near,
                    distance_far,
                ]
            } else {
                [
                    mesh.anisotropy,
                    mesh.anisotropy_rotation,
                    mesh.thickness,
                    mesh.attenuation_distance,
                ]
            },
            physical_params4: [
                mesh.clearcoat_normal_scale[0],
                mesh.clearcoat_normal_scale[1],
                mesh.light_map_intensity,
                if mesh.shading_model == ShadingModel::Matcap {
                    if mesh.matcap_map_is_srgb { 1.0 } else { 0.0 }
                } else if mesh.shading_model == ShadingModel::Toon {
                    if mesh.gradient_map_is_srgb { 1.0 } else { 0.0 }
                } else if mesh.specular_map.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
            attenuation_color: if mesh.shading_model == ShadingModel::Distance {
                [
                    distance_reference_position[0],
                    distance_reference_position[1],
                    distance_reference_position[2],
                    0.0,
                ]
            } else {
                [
                    mesh.attenuation_color[0],
                    mesh.attenuation_color[1],
                    mesh.attenuation_color[2],
                    mesh.dispersion,
                ]
            },
            physical_specular: [
                mesh.physical_specular_color[0],
                mesh.physical_specular_color[1],
                mesh.physical_specular_color[2],
                mesh.physical_specular_intensity,
            ],
            iridescence_params: [
                mesh.iridescence,
                mesh.iridescence_ior,
                mesh.iridescence_thickness_min,
                mesh.iridescence_thickness_max,
            ],
            lights,
        };
        let CachedUniformBindGroup {
            buffer: uniform_buffer,
            bind_group,
        } = self.dynamic_uniform_bind_group_for(dynamic_uniform_slots, uniform_slot, &uniforms);

        let (texture_bind_group, _mesh_texture) = match &mesh.texture {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::BaseColor,
                    &self.texture_layout,
                    "headless-three-renderer mesh texture",
                    "headless-three-renderer mesh texture bind group",
                    tex,
                );
                (bind_group, Some(texture))
            }
            None => (self.default_texture_bind_group.clone(), None),
        };

        let (normal_map_bind_group, _normal_map_texture) =
            match mesh.normal_map.as_ref().or(mesh.bump_map.as_ref()) {
                Some(tex) => {
                    let (bind_group, texture) = self.texture_bind_group_for(
                        TextureBindGroupKind::NormalOrBump,
                        &self.normal_map_layout,
                        "headless-three-renderer normal or bump map",
                        "headless-three-renderer normal or bump map bind group",
                        tex,
                    );
                    (bind_group, Some(texture))
                }
                None => (self.default_normal_map_bind_group.clone(), None),
            };

        let (mr_map_bind_group, _mr_map_texture) = match &mesh.metallic_roughness_texture {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::MetallicRoughness,
                    &self.mr_map_layout,
                    "headless-three-renderer metallic-roughness map",
                    "headless-three-renderer metallic-roughness bind group",
                    tex,
                );
                (bind_group, Some(texture))
            }
            None => (self.default_mr_map_bind_group.clone(), None),
        };

        let (emissive_map_bind_group, _emissive_map_texture) = match &mesh.emissive_map {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::Emissive,
                    &self.emissive_map_layout,
                    "headless-three-renderer emissive map",
                    "headless-three-renderer emissive map bind group",
                    tex,
                );
                (bind_group, Some(texture))
            }
            None => (self.default_emissive_map_bind_group.clone(), None),
        };

        let (
            ao_map_bind_group,
            _ao_map_texture,
            _light_map_texture,
            _alpha_map_texture,
            _physical_layers_map_texture,
            _physical_sheen_map_texture,
            _physical_specular_map_texture,
            _clearcoat_normal_map_texture,
        ) = if mesh.ao_map.is_some()
            || mesh.light_map.is_some()
            || mesh.specular_map.is_some()
            || mesh.alpha_map.is_some()
            || mesh.matcap_map.is_some()
            || mesh.gradient_map.is_some()
            || mesh.physical_maps.is_some()
            || mesh.clearcoat_normal_map.is_some()
        {
            let resources = self.ao_physical_bind_group_for(mesh);
            (
                resources.bind_group,
                resources.ao_texture,
                resources.light_texture,
                resources.alpha_texture,
                resources.physical_layers_texture,
                resources.physical_sheen_texture,
                resources.physical_specular_texture,
                resources.clearcoat_normal_texture,
            )
        } else {
            (
                self.default_ao_map_bind_group.clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        let pipeline_override = match mesh.custom_fragment_shader.as_deref() {
            Some(fragment_body) => {
                Some(self.create_custom_pipeline(mesh, fragment_body, settings.sample_count)?)
            }
            None if requires_pipeline_override(mesh, settings.sample_count) => {
                Some(self.create_state_override_pipeline(mesh, settings.sample_count))
            }
            None => None,
        };

        Ok(GpuMesh {
            vertex_buffer,
            index_buffer,
            bind_group,
            texture_bind_group,
            normal_map_bind_group,
            mr_map_bind_group,
            emissive_map_bind_group,
            ao_map_bind_group,
            pipeline_override,
            index_count,
            vertex_count,
            side: mesh.side,
            topology: mesh.topology,
            blend_constant: blend_constant(mesh.custom_blend),
            stencil_reference: if mesh.stencil_write {
                Some(mesh.stencil_ref)
            } else {
                None
            },
            cast_shadow: mesh.cast_shadow,
            _uniform_buffer: uniform_buffer,
            _texture: _mesh_texture,
            _normal_map: _normal_map_texture,
            _mr_map: _mr_map_texture,
            _emissive_map: _emissive_map_texture,
            _ao_map: _ao_map_texture,
            _light_map: _light_map_texture,
            _alpha_map: _alpha_map_texture,
            _physical_layers_map: _physical_layers_map_texture,
            _physical_sheen_map: _physical_sheen_map_texture,
            _physical_specular_map: _physical_specular_map_texture,
            _clearcoat_normal_map: _clearcoat_normal_map_texture,
        })
    }
}
