use super::*;

impl GpuRenderer {
    pub(super) fn write_texture_mip(
        &self,
        gpu_texture: &wgpu::Texture,
        mip_level: u32,
        array_layer: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: gpu_texture,
                mip_level,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(super) fn create_custom_pipeline(
        &self,
        mesh: &PreparedMesh,
        fragment_body: &str,
        sample_count: u32,
    ) -> Result<wgpu::RenderPipeline> {
        let key = CustomPipelineKey {
            state: StatePipelineKey::new(mesh, sample_count),
            fragment_body: fragment_body.to_owned(),
        };
        if let Some(pipeline) = self
            .custom_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(pipeline);
        }

        let source = custom_shader_source(fragment_body);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("headless-three-renderer custom material shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self.create_material_pipeline(
            &shader,
            mesh,
            sample_count,
            "headless-three-renderer custom material pipeline",
        );

        Ok(insert_bounded_cache(
            &mut self
                .custom_pipeline_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            pipeline,
            MAX_CUSTOM_PIPELINE_CACHE_ENTRIES,
        ))
    }

    pub(super) fn create_state_override_pipeline(
        &self,
        mesh: &PreparedMesh,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let key = StatePipelineKey::new(mesh, sample_count);
        if let Some(pipeline) = self
            .state_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return pipeline;
        }

        let pipeline = self.create_material_pipeline(
            &self.shader,
            mesh,
            sample_count,
            "headless-three-renderer material state override pipeline",
        );

        insert_bounded_cache(
            &mut self
                .state_pipeline_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            pipeline,
            MAX_STATE_PIPELINE_CACHE_ENTRIES,
        )
    }

    pub(super) fn create_material_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        mesh: &PreparedMesh,
        sample_count: u32,
        label: &'static str,
    ) -> wgpu::RenderPipeline {
        let color_targets = [Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: blend_state(
                mesh.blending,
                mesh.custom_blend,
                mesh.is_transparent,
                mesh.premultiplied_alpha,
            ),
            write_mask: if mesh.color_write {
                wgpu::ColorWrites::ALL
            } else {
                wgpu::ColorWrites::empty()
            },
        })];
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Vertex::layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology: mesh.topology.primitive(),
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: match mesh.topology {
                        Topology::Triangles => mesh.side.cull_mode(),
                        Topology::Lines | Topology::Points => None,
                    },
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(mesh.depth_write),
                    depth_compare: Some(depth_compare_function(mesh)),
                    stencil: stencil_state(mesh),
                    bias: depth_bias_state(mesh),
                }),
                multisample: multisample_state(sample_count, mesh.alpha_to_coverage),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &color_targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    pub(super) fn mesh_buffers_for(&self, mesh: &PreparedMesh) -> Result<CachedMeshBuffers> {
        if let Some(key) = mesh.native_mesh_key {
            if mesh.vertices.is_empty() {
                return self
                    .native_mesh_buffer_cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow!(
                            "native mesh buffer cache key {key} was referenced before a full mesh payload seeded it"
                        )
                    });
            }

            if let Some(buffers) = self
                .native_mesh_buffer_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&key)
                .cloned()
            {
                return Ok(buffers);
            }
        }

        let key = MeshBufferCacheKey::from_mesh(mesh);
        if let Some(buffers) = self
            .mesh_buffer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            if let Some(native_key) = mesh.native_mesh_key {
                insert_bounded_cache(
                    &mut self
                        .native_mesh_buffer_cache
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    native_key,
                    buffers.clone(),
                    MAX_NATIVE_MESH_BUFFER_CACHE_ENTRIES,
                );
            }
            return Ok(buffers);
        }

        let buffers = CachedMeshBuffers {
            vertex_buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("headless-three-renderer vertex buffer"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            index_buffer: mesh.indices.as_ref().map(|indices| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("headless-three-renderer index buffer"),
                        contents: bytemuck::cast_slice(indices),
                        usage: wgpu::BufferUsages::INDEX,
                    })
            }),
            vertex_count: mesh.vertices.len() as u32,
            index_count: mesh
                .indices
                .as_ref()
                .map(|indices| indices.len() as u32)
                .unwrap_or(0),
        };

        insert_bounded_cache(
            &mut self
                .mesh_buffer_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            buffers.clone(),
            MAX_MESH_BUFFER_CACHE_ENTRIES,
        );

        if let Some(native_key) = mesh.native_mesh_key {
            insert_bounded_cache(
                &mut self
                    .native_mesh_buffer_cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
                native_key,
                buffers.clone(),
                MAX_NATIVE_MESH_BUFFER_CACHE_ENTRIES,
            );
        }

        Ok(buffers)
    }

    pub(super) fn dynamic_uniform_bind_group_for(
        &self,
        slots: &mut Vec<CachedDynamicUniformBindGroup>,
        slot: usize,
        uniforms: &Uniforms,
    ) -> CachedUniformBindGroup {
        let key = UniformBindGroupKey::from_uniforms(uniforms);
        if let Some(cached) = slots.get_mut(slot) {
            if cached.key != key {
                self.queue
                    .write_buffer(&cached.buffer, 0, bytemuck::bytes_of(uniforms));
                cached.key = key;
            }
            return cached.as_uniform_bind_group();
        }

        debug_assert_eq!(slot, slots.len());
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer dynamic uniform buffer"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer dynamic bind group"),
            layout: &self.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let cached = CachedDynamicUniformBindGroup {
            key,
            buffer,
            bind_group,
        };
        slots.push(cached.clone());
        cached.as_uniform_bind_group()
    }

    pub(super) fn write_post_uniform_buffer(
        &self,
        slot: &mut Option<wgpu::Buffer>,
        uniforms: &PostUniforms,
    ) -> wgpu::Buffer {
        let bytes = bytemuck::bytes_of(uniforms);
        if let Some(buffer) = slot.as_ref() {
            self.queue.write_buffer(buffer, 0, bytes);
            return buffer.clone();
        }

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer post uniform buffer"),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        *slot = Some(buffer.clone());
        buffer
    }

    pub(super) fn post_bind_group_for(
        &self,
        source_size: wgpu::Extent3d,
        source_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = ScratchTextureKey::from_extent(source_size);
        if let Some(cached) = self
            .post_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return cached.bind_group();
        }

        let source_view = source_view.clone();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer post bind group"),
            layout: &self.post_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let cached = CachedPostBindGroup {
            source_view,
            bind_group,
        };
        insert_bounded_cache(
            &mut self
                .post_bind_group_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            cached,
            MAX_POST_BIND_GROUP_CACHE_ENTRIES,
        )
        .bind_group()
    }

    pub(super) fn ibl_bind_group_for(&self, ibl: &IblMaps) -> wgpu::BindGroup {
        let key = IblBindGroupKey::from_maps(ibl);
        if let Some(bind_group) = self
            .ibl_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return bind_group;
        }

        let bind_group = create_ibl_bind_group(
            &self.device,
            &self.queue,
            &self.ibl_layout,
            &self.sampler,
            ibl,
        );
        insert_bounded_cache(
            &mut self
                .ibl_bind_group_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            bind_group,
            MAX_IBL_BIND_GROUP_CACHE_ENTRIES,
        )
    }
}
