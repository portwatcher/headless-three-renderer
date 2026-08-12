use super::*;

enum RenderedFrame {
    Rgba(Vec<u8>),
    Texture(wgpu::Texture),
}

impl GpuRenderer {
    pub fn render(&self, scene: &RenderScene, camera: &Camera) -> Result<Vec<u8>> {
        let settings = RenderSettings::from_scene(scene, camera, self.device.limits())?;
        let meshes = prepare_meshes(scene)?;
        let rgba = self.render_rgba(&settings, &meshes)?;

        match settings.output_format {
            OutputFormat::Png => encode_png(&rgba, settings.width, settings.height),
            OutputFormat::Rgba => Ok(rgba),
        }
    }

    pub(super) fn render_rgba(
        &self,
        settings: &RenderSettings,
        meshes: &[PreparedMesh],
    ) -> Result<Vec<u8>> {
        match self.render_frame(settings, meshes, false, None)? {
            RenderedFrame::Rgba(rgba) => Ok(rgba),
            RenderedFrame::Texture(_) => unreachable!("CPU render returned a GPU texture"),
        }
    }

    pub fn render_gpu_frame(&self, scene: &RenderScene, camera: &Camera) -> Result<GpuFrame> {
        let capabilities = self.gpu_output_capabilities();
        if !capabilities.texture_supported {
            bail!(
                "native GPU texture output is unsupported: {}",
                capabilities.texture_reason.unwrap_or("unknown reason")
            );
        }
        let settings = RenderSettings::from_scene(scene, camera, self.device.limits())?;
        let meshes = prepare_meshes(scene)?;
        let texture = match self.render_frame(&settings, &meshes, true, None)? {
            RenderedFrame::Texture(texture) => texture,
            RenderedFrame::Rgba(_) => unreachable!("GPU render returned CPU pixels"),
        };
        Ok(GpuFrame::new(
            texture,
            self.backend,
            settings.width,
            settings.height,
        ))
    }

    pub(super) fn render_gpu_frame_into(
        &self,
        scene: &RenderScene,
        camera: &Camera,
        target: &wgpu::Texture,
    ) -> Result<()> {
        let settings = RenderSettings::from_scene(scene, camera, self.device.limits())?;
        let meshes = prepare_meshes(scene)?;
        match self.render_frame(&settings, &meshes, true, Some(target))? {
            RenderedFrame::Texture(_) => Ok(()),
            RenderedFrame::Rgba(_) => unreachable!("GPU render returned CPU pixels"),
        }
    }

    fn render_frame(
        &self,
        settings: &RenderSettings,
        meshes: &[PreparedMesh],
        native_output: bool,
        native_target: Option<&wgpu::Texture>,
    ) -> Result<RenderedFrame> {
        let texture_size = wgpu::Extent3d {
            width: settings.width,
            height: settings.height,
            depth_or_array_layers: 1,
        };

        let (color_texture, color_texture_guard) = self.cached_scratch_texture(
            &self.color_texture_cache,
            texture_size,
            "headless-three-renderer color texture",
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_color_texture = if settings.sample_count > 1 {
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless-three-renderer msaa color texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: settings.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_color_view = msaa_color_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let render_color_view = msaa_color_view.as_ref().unwrap_or(&color_view);

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer depth texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: settings.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut dynamic_uniform_guard = self
            .dynamic_uniform_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let gpu_meshes = meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| {
                self.upload_mesh(settings, mesh, index, &mut dynamic_uniform_guard)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scene_color_texture_guard = None;
        let mut post_texture_guard = None;

        let ibl_bind_group = match &settings.ibl {
            Some(ibl) => self.ibl_bind_group_for(ibl),
            None => self.default_ibl_bind_group.clone(),
        };

        // Optional shadow map pass: render a depth-only view from the
        // directional light's POV into a Depth32Float texture, then bind it
        // to the main pass.
        let (shadow_bind_group, _shadow_texture) = match &settings.shadow {
            Some(_shadow) => {
                let (bg, tex) = self.render_shadow_pass(settings, &gpu_meshes);
                (bg, Some(tex))
            }
            None => (self.default_shadow_bind_group.clone(), None),
        };

        let (opaque_order, transmissive_order, transparent_order) = partition_draw_order(meshes);

        let unpadded_bytes_per_row = settings.width * 4;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer_size = padded_bytes_per_row as u64 * settings.height as u64;
        if output_buffer_size > self.device.limits().max_buffer_size {
            bail!(
                "render output requires {} bytes, above device max_buffer_size {}",
                output_buffer_size,
                self.device.limits().max_buffer_size
            );
        }

        let mut readback_buffer_guard = None;
        let output_buffer = if native_output {
            None
        } else {
            let (buffer, guard) = self.cached_readback_buffer(output_buffer_size);
            readback_buffer_guard = Some(guard);
            Some(buffer)
        };
        let native_texture = native_output.then(|| {
            native_target.cloned().unwrap_or_else(|| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer leased output texture"),
                    size: texture_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer render encoder"),
            });
        let background_gpu = settings.background_texture.as_ref().map(|background| {
            self.upload_background(background, settings, settings.output_color_space)
        });
        let background_clear = wgpu::Color {
            r: settings.background[0] * f64::from(settings.background_intensity),
            g: settings.background[1] * f64::from(settings.background_intensity),
            b: settings.background[2] * f64::from(settings.background_intensity),
            a: settings.background[3],
        };

        if let Some(background) = &background_gpu {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_clear),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer background render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);
            pass.set_pipeline(self.background_pipeline_for(settings.sample_count));
            pass.set_bind_group(0, &background.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let color_load = match background_gpu.as_ref() {
                Some(_) => wgpu::LoadOp::Load,
                None => wgpu::LoadOp::Clear(background_clear),
            };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);

            // Opaque meshes first (with depth write)
            pass.set_bind_group(5, &ibl_bind_group, &[]);
            pass.set_bind_group(7, &shadow_bind_group, &[]);
            let mut current_pipeline: Option<PipelineKey> = None;
            for &i in &opaque_order {
                let mesh = &gpu_meshes[i];
                if let Some(pipeline) = &mesh.pipeline_override {
                    pass.set_pipeline(pipeline);
                    current_pipeline = None;
                } else {
                    let key = pipeline_key(mesh);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, false, settings.sample_count));
                        current_pipeline = Some(key);
                    }
                }
                if let Some(color) = mesh.blend_constant {
                    pass.set_blend_constant(color);
                }
                if let Some(reference) = mesh.stencil_reference {
                    pass.set_stencil_reference(reference);
                }
                draw_gpu_mesh(&mut pass, mesh);
            }
        }

        if !transmissive_order.is_empty() || !transparent_order.is_empty() {
            let scene_color_size = transmission_scene_color_size(settings);
            let (scene_color_texture, guard) = self.cached_scratch_texture(
                &self.scene_color_texture_cache,
                scene_color_size,
                "headless-three-renderer scene color texture",
                wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            scene_color_texture_guard = Some(guard);
            let scene_color_view =
                scene_color_texture.create_view(&wgpu::TextureViewDescriptor::default());
            if scene_color_size.width == texture_size.width
                && scene_color_size.height == texture_size.height
            {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &scene_color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    texture_size,
                );
            } else {
                self.blit_scene_color_for_transmission(
                    &mut encoder,
                    &color_view,
                    &scene_color_view,
                );
            }
            let shadow_view = match &_shadow_texture {
                Some(texture) => texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    array_layer_count: settings.shadow.as_ref().map(|s| s.layer_count),
                    ..Default::default()
                }),
                None => self
                    ._default_shadow_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        ..Default::default()
                    }),
            };
            let scene_shadow_bind_group =
                self.create_shadow_scene_bind_group(&shadow_view, &scene_color_view);

            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer transparent render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);

            pass.set_bind_group(5, &ibl_bind_group, &[]);
            pass.set_bind_group(7, &scene_shadow_bind_group, &[]);
            let mut current_pipeline: Option<PipelineKey> = None;
            for &i in transmissive_order.iter().chain(transparent_order.iter()) {
                let mesh = &gpu_meshes[i];
                if let Some(pipeline) = &mesh.pipeline_override {
                    pass.set_pipeline(pipeline);
                    current_pipeline = None;
                } else {
                    let key = pipeline_key(mesh);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, true, settings.sample_count));
                        current_pipeline = Some(key);
                    }
                }
                if let Some(color) = mesh.blend_constant {
                    pass.set_blend_constant(color);
                }
                if let Some(reference) = mesh.stencil_reference {
                    pass.set_stencil_reference(reference);
                }
                draw_gpu_mesh(&mut pass, mesh);
            }
        }

        let mut post_uniform_buffer_guard = None;
        if settings.post_processing.active {
            let (post_texture, guard) = self.cached_scratch_texture(
                &self.post_texture_cache,
                texture_size,
                "headless-three-renderer post color texture",
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            post_texture_guard = Some(guard);
            let post_view = post_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let post_uniforms = post_uniforms(settings.post_processing);
            let mut guard = self
                .post_uniform_buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let post_uniform_buffer = self.write_post_uniform_buffer(&mut guard, &post_uniforms);
            let post_bind_group =
                self.post_bind_group_for(texture_size, &color_view, &post_uniform_buffer);
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &post_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer post pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &post_bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

            copy_texture_to_render_output(
                &mut encoder,
                &post_texture,
                output_buffer.as_ref(),
                native_texture.as_ref(),
                padded_bytes_per_row,
                settings.height,
                texture_size,
            );
            post_uniform_buffer_guard = Some(guard);
        } else {
            copy_texture_to_render_output(
                &mut encoder,
                &color_texture,
                output_buffer.as_ref(),
                native_texture.as_ref(),
                padded_bytes_per_row,
                settings.height,
                texture_size,
            );
        }

        self.queue.submit([encoder.finish()]);

        let mapped = output_buffer.as_ref().map(|output_buffer| {
            let buffer_slice = output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            (buffer_slice, receiver)
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for GPU frame completion")?;

        let rgba = if let Some((buffer_slice, receiver)) = mapped {
            receiver
                .recv()
                .context("GPU readback callback was not delivered")?
                .context("failed to map GPU readback buffer")?;
            let padded_data = buffer_slice.get_mapped_range();
            let mut rgba = vec![0; (settings.width * settings.height * 4) as usize];
            let unpadded = unpadded_bytes_per_row as usize;
            let padded = padded_bytes_per_row as usize;
            for row in 0..settings.height as usize {
                let src_start = row * padded;
                let dst_start = row * unpadded;
                rgba[dst_start..dst_start + unpadded]
                    .copy_from_slice(&padded_data[src_start..src_start + unpadded]);
            }
            drop(padded_data);
            output_buffer.as_ref().unwrap().unmap();
            Some(rgba)
        } else {
            None
        };
        drop(post_uniform_buffer_guard);
        drop(post_texture_guard);
        drop(scene_color_texture_guard);
        drop(color_texture_guard);
        drop(readback_buffer_guard);
        dynamic_uniform_guard.truncate(meshes.len());
        drop(dynamic_uniform_guard);

        if let Some(rgba) = rgba {
            Ok(RenderedFrame::Rgba(rgba))
        } else {
            Ok(RenderedFrame::Texture(
                native_texture.context("native output texture was not created")?,
            ))
        }
    }

    pub(super) fn cached_scratch_texture<'a>(
        &self,
        cache: &'a Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
        size: wgpu::Extent3d,
        label: &'static str,
        usage: wgpu::TextureUsages,
    ) -> (
        wgpu::Texture,
        MutexGuard<'a, HashMap<ScratchTextureKey, wgpu::Texture>>,
    ) {
        let key = ScratchTextureKey::from_extent(size);
        let mut guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let texture = if let Some(texture) = guard.get(&key) {
            texture.clone()
        } else {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage,
                view_formats: &[],
            });
            insert_bounded_cache(&mut guard, key, texture, MAX_SCRATCH_TEXTURE_CACHE_ENTRIES)
        };

        (texture, guard)
    }

    pub(super) fn cached_readback_buffer(
        &self,
        size: u64,
    ) -> (
        wgpu::Buffer,
        MutexGuard<'_, HashMap<ReadbackBufferKey, wgpu::Buffer>>,
    ) {
        let key = ReadbackBufferKey::from_size(size);
        let mut guard = self
            .readback_buffer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let buffer = if let Some(buffer) = guard.get(&key) {
            buffer.clone()
        } else {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless-three-renderer readback buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            insert_bounded_cache(&mut guard, key, buffer, MAX_READBACK_BUFFER_CACHE_ENTRIES)
        };

        (buffer, guard)
    }
}
