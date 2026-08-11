use super::*;

#[allow(dead_code)] // Texture owners keep bind-group resources alive during initialization.
pub(super) struct RendererDefaults {
    pub(super) sampler: wgpu::Sampler,
    pub(super) shadow_sampler: wgpu::Sampler,
    pub(super) default_texture: wgpu::Texture,
    pub(super) default_texture_view: wgpu::TextureView,
    pub(super) default_texture_bind_group: wgpu::BindGroup,
    pub(super) default_normal_map: wgpu::Texture,
    pub(super) default_normal_map_view: wgpu::TextureView,
    pub(super) default_normal_map_bind_group: wgpu::BindGroup,
    pub(super) default_mr_map_bind_group: wgpu::BindGroup,
    pub(super) default_emissive_map: wgpu::Texture,
    pub(super) default_emissive_map_bind_group: wgpu::BindGroup,
    pub(super) default_ibl_bind_group: wgpu::BindGroup,
    pub(super) default_physical_layers_texture: wgpu::Texture,
    pub(super) default_ao_map_bind_group: wgpu::BindGroup,
}

pub(super) fn create_renderer_defaults(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layouts: &RendererLayouts,
) -> RendererDefaults {
    let texture_layout = &layouts.texture_layout;
    let normal_map_layout = &layouts.normal_map_layout;
    let mr_map_layout = &layouts.mr_map_layout;
    let emissive_map_layout = &layouts.emissive_map_layout;
    let ao_map_layout = &layouts.ao_map_layout;
    let ibl_layout = &layouts.ibl_layout;
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("headless-three-renderer sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("headless-three-renderer shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    let default_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-three-renderer default texture"),
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
            texture: &default_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
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
    let default_texture_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let default_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default texture bind group"),
        layout: &texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Default normal map: flat normal (0, 0, 1) encoded as (128, 128, 255)
    let default_normal_map = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-three-renderer default normal map"),
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
            texture: &default_normal_map,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[128u8, 128, 255, 255],
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
    let default_normal_map_view =
        default_normal_map.create_view(&wgpu::TextureViewDescriptor::default());
    let default_normal_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default normal map bind group"),
        layout: &normal_map_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_normal_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Default metallic-roughness map: white (1,1,1,1) so that
    // metallic = uniform.metallic * 1.0 and roughness = uniform.roughness * 1.0
    let default_mr_map_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let default_mr_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default metallic-roughness bind group"),
        layout: &mr_map_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_mr_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Default emissive map: black (0,0,0,255) so that emissive contribution is zero
    // when no emissive map is provided
    let default_emissive_map = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-three-renderer default emissive map"),
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
            texture: &default_emissive_map,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
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
    let default_emissive_map_view =
        default_emissive_map.create_view(&wgpu::TextureViewDescriptor::default());
    let default_emissive_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default emissive map bind group"),
        layout: &emissive_map_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_emissive_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Default IBL: 1x1 black cubemaps (no env map contribution)
    let default_ibl_bind_group =
        create_default_ibl_bind_group(&device, &queue, &ibl_layout, &sampler);

    // Default physical layers: layer 0 is neutral scalar/specular data, layer 1 is
    // the default +X anisotropy direction with full strength, and layer 2 is
    // neutral iridescence factor/thickness data.
    let default_physical_layers_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-three-renderer default physical layers map"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 3,
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
            texture: &default_physical_layers_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &default_physical_layers_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 128, 255, 255],
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &default_physical_layers_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
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
    let default_physical_layers_view =
        default_physical_layers_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

    // Default AO map: reuse the 1x1 white default texture; red channel = 1.0
    // means full illumination (no occlusion).
    let default_ao_map_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let default_ao_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default ao and physical maps bind group"),
        layout: &ao_map_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&default_physical_layers_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&default_normal_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    RendererDefaults {
        sampler,
        shadow_sampler,
        default_texture,
        default_texture_view,
        default_texture_bind_group,
        default_normal_map,
        default_normal_map_view,
        default_normal_map_bind_group,
        default_mr_map_bind_group,
        default_emissive_map,
        default_emissive_map_bind_group,
        default_ibl_bind_group,
        default_physical_layers_texture,
        default_ao_map_bind_group,
    }
}
