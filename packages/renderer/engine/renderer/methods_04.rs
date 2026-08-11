use super::*;

impl GpuRenderer {
    /// Render the scene's shadow casters into a shared depth-only texture array
    /// and return a bind group referencing it.
    pub(super) fn render_shadow_pass(
        &self,
        settings: &RenderSettings,
        gpu_meshes: &[GpuMesh],
    ) -> (wgpu::BindGroup, wgpu::Texture) {
        let shadow_maps = settings
            .shadow
            .as_ref()
            .expect("render_shadow_pass requires configured shadow maps");
        let shadow_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer shadow map"),
            size: wgpu::Extent3d {
                width: shadow_maps.map_width,
                height: shadow_maps.map_height,
                depth_or_array_layers: shadow_maps.layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(shadow_maps.layer_count),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer shadow encoder"),
            });

        for caster in &shadow_maps.casters {
            for local_layer in 0..caster.layer_count {
                let layer = caster.layer_base + local_layer;
                let layer_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("headless-three-renderer shadow pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &layer_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                for mesh in gpu_meshes.iter() {
                    if !mesh.cast_shadow {
                        continue;
                    }
                    match mesh.topology {
                        Topology::Triangles => {
                            pass.set_pipeline(&self.shadow_pipelines[layer as usize]);
                        }
                        Topology::Lines => {
                            pass.set_pipeline(&self.line_shadow_pipelines[layer as usize]);
                        }
                        Topology::Points => continue,
                    }
                    pass.set_bind_group(0, &mesh.bind_group, &[]);
                    pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
                    pass.set_bind_group(2, &mesh.normal_map_bind_group, &[]);
                    pass.set_bind_group(3, &mesh.mr_map_bind_group, &[]);
                    pass.set_bind_group(4, &mesh.emissive_map_bind_group, &[]);
                    pass.set_bind_group(5, &self.default_ibl_bind_group, &[]);
                    pass.set_bind_group(6, &mesh.ao_map_bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    if let Some(index_buffer) = &mesh.index_buffer {
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    } else {
                        pass.draw(0..mesh.vertex_count, 0..1);
                    }
                }
            }
        }

        self.queue.submit([encoder.finish()]);

        let default_scene_view = self
            ._default_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer shadow bind group"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        (bind_group, shadow_texture)
    }

    pub(super) fn create_shadow_scene_bind_group(
        &self,
        shadow_view: &wgpu::TextureView,
        scene_color_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer shadow scene-color bind group"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    pub(super) fn blit_scene_color_for_transmission(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_view: &wgpu::TextureView,
        destination_view: &wgpu::TextureView,
    ) {
        let post_uniforms = post_uniforms(PostProcessingSettings {
            active: false,
            exposure: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            vignette: 0.0,
            grayscale: 0.0,
            invert: 0.0,
        });
        let mut guard = self
            .post_uniform_buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let post_uniform_buffer = self.write_post_uniform_buffer(&mut guard, &post_uniforms);
        let post_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer transmission scene-color blit bind group"),
            layout: &self.post_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: destination_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless-three-renderer transmission scene-color blit pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &post_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn sampler_for_texture(
        &self,
        wrap_s: WrapMode,
        wrap_t: WrapMode,
        mag_filter: TextureFilter,
        min_filter: TextureFilter,
        mipmap_mode: MipmapFilter,
        anisotropy: u16,
    ) -> wgpu::Sampler {
        let key = SamplerKey::new(
            wrap_s,
            wrap_t,
            mag_filter,
            min_filter,
            mipmap_mode,
            anisotropy,
        );
        if key.is_default() {
            return self.sampler.clone();
        }

        if let Some(sampler) = self
            .sampler_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return sampler;
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer per-mesh sampler"),
            address_mode_u: key.wrap_s.to_address_mode(),
            address_mode_v: key.wrap_t.to_address_mode(),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: key.mag_filter.to_filter_mode(),
            min_filter: key.min_filter.to_filter_mode(),
            mipmap_filter: key.mipmap_filter.to_mipmap_filter_mode(),
            lod_max_clamp: key.lod_max_clamp(),
            anisotropy_clamp: key.anisotropy_clamp,
            ..Default::default()
        });

        insert_bounded_cache(
            &mut self
                .sampler_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            sampler,
            MAX_SAMPLER_CACHE_ENTRIES,
        )
    }

    pub(super) fn sampler_for_settings(&self, settings: TextureSamplerSettings) -> wgpu::Sampler {
        self.sampler_for_texture(
            settings.wrap_s,
            settings.wrap_t,
            settings.mag_filter,
            settings.min_filter,
            settings.mipmap_filter,
            settings.anisotropy,
        )
    }

    pub(super) fn upload_background(
        &self,
        background: &BackgroundTexture,
        settings: &RenderSettings,
        output_color_space: OutputColorSpace,
    ) -> GpuBackground<'_> {
        let gpu_texture = self.upload_texture(
            "headless-three-renderer scene background texture",
            &background.texture,
        );
        let background_flags = if background.is_srgb { 1.0 } else { 0.0 }
            + if output_color_space.is_linear() {
                2.0
            } else {
                0.0
            }
            + if background.mapping == BackgroundTextureMapping::Equirectangular {
                4.0
            } else {
                0.0
            }
            + background.blurriness * 0.25;
        let uniforms = BackgroundUniforms {
            transform1: [
                background.transform[0],
                background.transform[1],
                background.transform[2],
                background.intensity,
            ],
            transform2: [
                background.transform[3],
                background.transform[4],
                background.transform[5],
                background_flags,
            ],
            inverse_view_projection: settings.view_projection.inverse().to_cols_array_2d(),
            camera_params: [
                settings.camera_pos.x,
                settings.camera_pos.y,
                settings.camera_pos.z,
                0.0,
            ],
            rotation1: background.rotation[0],
            rotation2: background.rotation[1],
            rotation3: background.rotation[2],
        };
        let key = BackgroundBindGroupKey::new(&background.texture);
        let mut cache_guard = self
            .background_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache_guard.get(&key).cloned() {
            self.queue
                .write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            return GpuBackground {
                bind_group: cached.bind_group,
                _texture: gpu_texture,
                _uniform_buffer: cached.uniform_buffer,
                _cache_guard: cache_guard,
            };
        }

        let texture_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.sampler_for_texture(
            background.texture.wrap_s,
            background.texture.wrap_t,
            background.texture.mag_filter,
            background.texture.min_filter,
            background.texture.mipmap_filter,
            background.texture.anisotropy,
        );
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer background uniform buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer background bind group"),
            layout: &self.background_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let cached = CachedBackgroundBindGroup {
            uniform_buffer,
            bind_group,
        };
        let cached = insert_bounded_cache(
            &mut cache_guard,
            key,
            cached,
            MAX_BACKGROUND_BIND_GROUP_CACHE_ENTRIES,
        );
        GpuBackground {
            bind_group: cached.bind_group,
            _texture: gpu_texture,
            _uniform_buffer: cached.uniform_buffer,
            _cache_guard: cache_guard,
        }
    }

    pub(super) fn upload_texture(
        &self,
        label: &'static str,
        tex: &PreparedTexture,
    ) -> wgpu::Texture {
        let key = TextureCacheKey::from_texture(tex);
        if let Some(texture) = self
            .texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return texture;
        }

        let texture = self.upload_texture_uncached(label, tex);
        insert_bounded_cache(
            &mut self
                .texture_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            texture,
            MAX_TEXTURE_CACHE_ENTRIES,
        )
    }

    pub(super) fn upload_texture_uncached(
        &self,
        label: &'static str,
        tex: &PreparedTexture,
    ) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        };
        let mip_level_count = texture_upload_mip_level_count(tex);
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.write_prepared_texture_mip_chain(&gpu_texture, 0, tex, mip_level_count);
        gpu_texture
    }

    pub(super) fn texture_bind_group_for(
        &self,
        kind: TextureBindGroupKind,
        layout: &wgpu::BindGroupLayout,
        texture_label: &'static str,
        bind_group_label: &'static str,
        tex: &PreparedTexture,
    ) -> (wgpu::BindGroup, wgpu::Texture) {
        let gpu_texture = self.upload_texture(texture_label, tex);
        let key = TextureBindGroupKey::new(kind, tex);
        if let Some(bind_group) = self
            .texture_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return (bind_group, gpu_texture);
        }

        let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler_for_tex = self.sampler_for_texture(
            tex.wrap_s,
            tex.wrap_t,
            tex.mag_filter,
            tex.min_filter,
            tex.mipmap_filter,
            tex.anisotropy,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(bind_group_label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_for_tex),
                },
            ],
        });

        let bind_group = insert_bounded_cache(
            &mut self
                .texture_bind_group_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            bind_group,
            MAX_TEXTURE_BIND_GROUP_CACHE_ENTRIES,
        );
        (bind_group, gpu_texture)
    }
}
