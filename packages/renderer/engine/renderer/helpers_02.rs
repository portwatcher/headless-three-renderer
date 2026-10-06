use super::*;

impl GpuRenderer {
    pub(super) fn background_pipeline_for(&self, sample_count: u32) -> &wgpu::RenderPipeline {
        if sample_count == 4 {
            &self.background_pipeline_msaa4
        } else {
            &self.background_pipeline
        }
    }

    pub(super) fn pipeline_for(
        &self,
        key: PipelineKey,
        transparent: bool,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        let msaa4 = sample_count == 4;
        match key {
            PipelineKey::Tri(side) => {
                let idx = side_index(side);
                if transparent && msaa4 {
                    &self.transparent_pipelines_msaa4[idx]
                } else if transparent {
                    &self.transparent_pipelines[idx]
                } else if msaa4 {
                    &self.pipelines_msaa4[idx]
                } else {
                    &self.pipelines[idx]
                }
            }
            PipelineKey::Line if msaa4 => {
                &self.line_pipelines_msaa4[if transparent { 1 } else { 0 }]
            }
            PipelineKey::Line => &self.line_pipelines[if transparent { 1 } else { 0 }],
            PipelineKey::Point if msaa4 => {
                &self.point_pipelines_msaa4[if transparent { 1 } else { 0 }]
            }
            PipelineKey::Point => &self.point_pipelines[if transparent { 1 } else { 0 }],
        }
    }
}

pub(super) fn partition_draw_order(
    meshes: &[PreparedMesh],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut opaque = Vec::new();
    let mut transmissive = Vec::new();
    let mut transparent = Vec::new();

    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.transmission > 0.0001 {
            transmissive.push(i);
        } else if mesh.is_transparent {
            transparent.push(i);
        } else {
            opaque.push(i);
        }
    }

    opaque.sort_by(|&a, &b| compare_opaque_meshes(&meshes[a], &meshes[b]));

    // Sort transparent meshes back-to-front (farthest first)
    transmissive.sort_by(|&a, &b| compare_transparent_meshes(&meshes[a], &meshes[b]));
    transparent.sort_by(|&a, &b| compare_transparent_meshes(&meshes[a], &meshes[b]));

    (opaque, transmissive, transparent)
}

pub(super) fn compare_opaque_meshes(a: &PreparedMesh, b: &PreparedMesh) -> std::cmp::Ordering {
    compare_f32(a.group_order, b.group_order)
        .then_with(|| compare_f32(a.render_order, b.render_order))
        .then_with(|| a.material_sort_key.cmp(&b.material_sort_key))
        .then_with(|| a.material_variant.cmp(&b.material_variant))
        .then_with(|| compare_f32(a.sort_z, b.sort_z))
        .then_with(|| a.sort_index.cmp(&b.sort_index))
}

pub(super) fn compare_transparent_meshes(a: &PreparedMesh, b: &PreparedMesh) -> std::cmp::Ordering {
    compare_f32(a.group_order, b.group_order)
        .then_with(|| compare_f32(a.render_order, b.render_order))
        .then_with(|| compare_f32(b.sort_z, a.sort_z))
        .then_with(|| a.sort_index.cmp(&b.sort_index))
}

pub(super) fn compare_f32(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

pub(super) fn draw_gpu_mesh(pass: &mut wgpu::RenderPass, mesh: &GpuMesh) {
    pass.set_bind_group(0, &mesh.bind_group, &[]);
    pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
    pass.set_bind_group(2, &mesh.normal_map_bind_group, &[]);
    pass.set_bind_group(3, &mesh.mr_map_bind_group, &[]);
    pass.set_bind_group(4, &mesh.emissive_map_bind_group, &[]);
    // bind group 5 (IBL) is set once per pass, not per mesh
    pass.set_bind_group(6, &mesh.ao_map_bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    if let Some(index_buffer) = &mesh.index_buffer {
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
    } else {
        pass.draw(0..mesh.vertex_count, 0..1);
    }
}

pub(super) fn map_transform_rows(mesh: &PreparedMesh) -> [[f32; 4]; 12] {
    let transforms = [
        if mesh.normal_map.is_some() {
            mesh.normal_map_transform
        } else {
            mesh.bump_map_transform
        },
        mesh.metallic_roughness_texture_transform,
        mesh.emissive_map_transform,
        mesh.ao_map_transform,
        mesh.light_map_transform,
        mesh.specular_map_transform,
    ];
    let mut rows = [[0.0; 4]; 12];
    for (index, transform) in transforms.iter().enumerate() {
        let row = index * 2;
        rows[row] = [transform[0], transform[1], transform[2], 0.0];
        rows[row + 1] = [transform[3], transform[4], transform[5], 0.0];
    }
    rows[1][3] = if mesh.normal_map.is_some() {
        if mesh.normal_map_uses_uv2 { 1.0 } else { 0.0 }
    } else if mesh.bump_map.is_some() {
        if mesh.bump_map_uses_uv2 { 1.0 } else { 0.0 }
    } else {
        0.0
    };
    rows[2][3] = if mesh.metallic_roughness_texture_is_srgb {
        1.0
    } else {
        0.0
    };
    rows[3][3] = if mesh.metallic_roughness_texture_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[4][3] = if mesh.emissive_map_is_srgb { 1.0 } else { 0.0 };
    rows[5][3] = if mesh.emissive_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[6][3] = if mesh.ao_map_is_srgb { 1.0 } else { 0.0 };
    rows[7][3] = if mesh.ao_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[8][3] = if mesh.light_map_is_srgb { 1.0 } else { 0.0 };
    rows[9][3] = if mesh.light_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[10][3] = if mesh.specular_map_is_srgb { 1.0 } else { 0.0 };
    rows[11][3] = if mesh.specular_map_uses_uv2 { 1.0 } else { 0.0 };
    rows
}

pub(super) fn physical_map_transform_rows(mesh: &PreparedMesh) -> [[f32; 4]; 24] {
    let transforms = [
        mesh.clearcoat_map_transform,
        mesh.clearcoat_roughness_map_transform,
        mesh.clearcoat_normal_map_transform,
        if matches!(
            mesh.shading_model,
            ShadingModel::Matcap | ShadingModel::Mtoon
        ) {
            mesh.matcap_map_transform
        } else {
            mesh.sheen_color_map_transform
        },
        mesh.sheen_roughness_map_transform,
        mesh.anisotropy_map_transform,
        mesh.transmission_map_transform,
        mesh.thickness_map_transform,
        mesh.specular_color_map_transform,
        mesh.specular_intensity_map_transform,
        mesh.iridescence_map_transform,
        mesh.iridescence_thickness_map_transform,
    ];
    let mut rows = [[0.0; 4]; 24];
    for (index, transform) in transforms.iter().enumerate() {
        let row = index * 2;
        rows[row] = [transform[0], transform[1], transform[2], 0.0];
        rows[row + 1] = [transform[3], transform[4], transform[5], 0.0];
    }
    if matches!(
        mesh.shading_model,
        ShadingModel::Matcap | ShadingModel::Mtoon
    ) {
        rows[7][3] = if mesh.matcap_map_uses_uv2 { 1.0 } else { 0.0 };
    } else {
        rows[7][3] = if mesh.sheen_color_map_uses_uv2 {
            1.0
        } else {
            0.0
        };
    }
    rows[1][3] = if mesh.clearcoat_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[3][3] = if mesh.clearcoat_roughness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[5][3] = if mesh.clearcoat_normal_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[9][3] = if mesh.sheen_roughness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[11][3] = if mesh.anisotropy_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[13][3] = if mesh.transmission_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[15][3] = if mesh.thickness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[17][3] = if mesh.specular_color_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[19][3] = if mesh.specular_intensity_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[21][3] = if mesh.iridescence_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[23][3] = if mesh.iridescence_thickness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows
}

pub(super) fn light_probe_rows(settings: &RenderSettings) -> [[f32; 4]; 9] {
    let mut rows = [[0.0; 4]; 9];
    for (index, coefficient) in settings.light_probe.iter().enumerate() {
        rows[index] = [coefficient[0], coefficient[1], coefficient[2], 0.0];
    }
    rows
}

pub(super) fn post_uniforms(settings: PostProcessingSettings) -> PostUniforms {
    PostUniforms {
        params1: [
            settings.exposure,
            settings.contrast,
            settings.saturation,
            settings.vignette,
        ],
        params2: [settings.grayscale, settings.invert, 0.0, 0.0],
    }
}

pub(super) fn transmission_scene_color_size(settings: &RenderSettings) -> wgpu::Extent3d {
    let scale = settings.transmission_resolution_scale;
    wgpu::Extent3d {
        width: ((settings.width as f32 * scale).round() as u32).max(1),
        height: ((settings.height as f32 * scale).round() as u32).max(1),
        depth_or_array_layers: 1,
    }
}

pub(super) fn copy_texture_to_render_output(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    output_buffer: Option<&wgpu::Buffer>,
    native_texture: Option<&wgpu::Texture>,
    padded_bytes_per_row: u32,
    height: u32,
    texture_size: wgpu::Extent3d,
) {
    if let Some(output_buffer) = output_buffer {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            texture_size,
        );
    }
    if let Some(native_texture) = native_texture {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: native_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_size,
        );
    }
}

pub(super) fn create_default_ibl_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    // Black 1x1 environment cube and CubeUV atlas for meshes without image-based lighting.
    let black = [0u8; 8];
    let env_cube = create_rgba16f_cubemap(device, queue, 1, &[&black[..]; 6]);
    let env_cube_view = env_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let cube_uv = create_rgba16f_texture(device, queue, "default cube uv atlas", 1, 1, &black);
    let cube_uv_view = cube_uv.create_view(&wgpu::TextureViewDescriptor::default());
    create_ibl_bind_group_from_views(device, layout, sampler, &env_cube_view, &cube_uv_view)
}

pub(super) fn create_ibl_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ibl: &IblMaps,
) -> wgpu::BindGroup {
    let faces = ibl
        .env_cube_faces
        .iter()
        .map(|face| face.as_slice())
        .collect::<Vec<_>>();
    let env_cube = create_rgba16f_cubemap(device, queue, ibl.env_cube_size, &faces);
    let env_cube_view = env_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let cube_uv = create_rgba16f_texture(
        device,
        queue,
        "pmrem cube uv atlas",
        ibl.cube_uv_width,
        ibl.cube_uv_height,
        &ibl.cube_uv,
    );
    let cube_uv_view = cube_uv.create_view(&wgpu::TextureViewDescriptor::default());
    create_ibl_bind_group_from_views(device, layout, sampler, &env_cube_view, &cube_uv_view)
}

fn create_ibl_bind_group_from_views(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    env_cube_view: &wgpu::TextureView,
    cube_uv_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(env_cube_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(cube_uv_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_rgba16f_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * width),
            rows_per_image: Some(height),
        },
        size,
    );
    texture
}

pub(super) fn create_rgba16f_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    faces: &[&[u8]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("environment cubemap"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (face, data) in faces.iter().enumerate().take(6) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}
