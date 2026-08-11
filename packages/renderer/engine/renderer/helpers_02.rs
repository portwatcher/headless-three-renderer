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
        if mesh.shading_model == ShadingModel::Matcap {
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
    if mesh.shading_model == ShadingModel::Matcap {
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
    // 1x1 black cubemap for irradiance and prefilter
    let black_cube = create_cubemap(device, queue, 1, 1, &[&[0u8, 0, 0, 255] as &[u8]; 6]);
    let irradiance_view = black_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let prefilter_view = black_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // 1x1 BRDF LUT with (0, 0, 0, 255)
    let brdf_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("default brdf lut"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &brdf_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8, 0, 0, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("default ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&prefilter_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(super) fn create_ibl_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ibl: &IblMaps,
) -> wgpu::BindGroup {
    // Irradiance cubemap
    let irradiance_tex = create_cubemap(
        device,
        queue,
        ibl.irradiance_size,
        1,
        &ibl.irradiance_faces
            .iter()
            .map(|f| f.as_slice())
            .collect::<Vec<_>>(),
    );
    let irradiance_view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // Prefiltered specular cubemap with mip levels
    let prefilter_tex = create_cubemap_with_mips(
        device,
        queue,
        ibl.prefilter_base_size,
        ibl.prefilter_mip_levels,
        &ibl.prefilter_faces,
    );
    let prefilter_view = prefilter_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // BRDF LUT
    let brdf_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf lut"),
        size: wgpu::Extent3d {
            width: ibl.brdf_lut_size,
            height: ibl.brdf_lut_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &brdf_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &ibl.brdf_lut,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * ibl.brdf_lut_size),
            rows_per_image: Some(ibl.brdf_lut_size),
        },
        wgpu::Extent3d {
            width: ibl.brdf_lut_size,
            height: ibl.brdf_lut_size,
            depth_or_array_layers: 1,
        },
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&prefilter_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(super) fn create_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    mip_levels: u32,
    faces: &[&[u8]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cubemap"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (face, data) in faces.iter().enumerate() {
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
                bytes_per_row: Some(4 * size),
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
