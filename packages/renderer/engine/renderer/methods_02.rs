use super::*;

impl GpuRenderer {
    pub(super) async fn new_async() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = match instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .await
{
    Ok(adapter) => adapter,
    Err(first_error) => instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: true,
            compatible_surface: None,
        })
        .await
        .map_err(|fallback_error| {
            anyhow!(
                "failed to find a wgpu adapter: {first_error}; fallback adapter also failed: {fallback_error}"
            )
        })?,
};

        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let backend = adapter.get_info().backend;
        let adapter_specific_storage = adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        let media_i420_planes_supported = adapter_specific_storage
            && storage_format_supported(&adapter, wgpu::TextureFormat::R8Unorm);
        let media_nv12_planes_supported = media_i420_planes_supported
            && storage_format_supported(&adapter, wgpu::TextureFormat::Rg8Unorm);
        let media_p010_planes_supported = adapter_specific_storage
            && adapter
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
            && storage_format_supported(&adapter, wgpu::TextureFormat::R16Unorm)
            && storage_format_supported(&adapter, wgpu::TextureFormat::Rg16Unorm);
        let optional_media_features = adapter.features()
            & (wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        // Raise the bind group count: we use 8 (uniforms + 5 texture slots +
        // IBL + shadow map), which is the WebGPU spec default.
        let required_limits = wgpu::Limits {
            max_bind_groups: 8,
            ..required_limits
        };
        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("headless-three-renderer device"),
            required_features: optional_media_features,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue, encoder_prerequisites) =
            linux_encoder::open_renderer_device(&adapter, &device_descriptor)
                .await
                .context("failed to create renderer device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer post shader"),
            source: wgpu::ShaderSource::Wgsl(POST_SHADER.into()),
        });
        let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer background shader"),
            source: wgpu::ShaderSource::Wgsl(BACKGROUND_SHADER.into()),
        });

        let layouts = create_renderer_layouts(&device);
        let defaults = create_renderer_defaults(&device, &queue, &layouts);
        let pipelines = create_renderer_pipelines(
            &device,
            &shader,
            &post_shader,
            &background_shader,
            &layouts,
            &defaults,
        );
        let RendererLayouts {
            uniform_layout,
            texture_layout,
            normal_map_layout,
            mr_map_layout,
            emissive_map_layout,
            ao_map_layout,
            shadow_layout,
            ibl_layout,
            post_layout,
            background_layout,
        } = layouts;
        let RendererDefaults {
            sampler,
            shadow_sampler,
            default_texture,
            default_texture_view: _,
            default_texture_bind_group,
            default_normal_map,
            default_normal_map_view: _,
            default_normal_map_bind_group,
            default_mr_map_bind_group,
            default_emissive_map: _,
            default_emissive_map_bind_group,
            default_ibl_bind_group,
            default_physical_layers_texture,
            default_ao_map_bind_group,
        } = defaults;
        let RendererPipelines {
            pipeline_layout,
            default_shadow_texture,
            default_shadow_bind_group,
            shadow_pipelines,
            line_shadow_pipelines,
            pipelines,
            pipelines_msaa4,
            transparent_pipelines,
            transparent_pipelines_msaa4,
            line_pipelines,
            line_pipelines_msaa4,
            point_pipelines,
            point_pipelines_msaa4,
            post_pipeline,
            background_pipeline,
            background_pipeline_msaa4,
        } = pipelines;
        Ok(Self {
            device,
            queue,
            backend,
            media_nv12_planes_supported,
            media_p010_planes_supported,
            media_i420_planes_supported,
            encoder_prerequisites,
            media_worker: std::sync::Arc::new(MediaWorker::new()?),
            shader,
            pipelines,
            pipelines_msaa4,
            transparent_pipelines,
            transparent_pipelines_msaa4,
            line_pipelines,
            line_pipelines_msaa4,
            point_pipelines,
            point_pipelines_msaa4,
            pipeline_layout,
            post_layout,
            background_layout,
            post_pipeline,
            background_pipeline,
            background_pipeline_msaa4,
            uniform_layout,
            texture_layout,
            normal_map_layout,
            mr_map_layout,
            emissive_map_layout,
            ibl_layout,
            ao_map_layout,
            shadow_layout,
            shadow_pipelines,
            line_shadow_pipelines,
            sampler,
            sampler_cache: Mutex::new(HashMap::new()),
            texture_cache: Mutex::new(HashMap::new()),
            physical_layers_texture_cache: Mutex::new(HashMap::new()),
            texture_bind_group_cache: Mutex::new(HashMap::new()),
            ao_physical_bind_group_cache: Mutex::new(HashMap::new()),
            background_bind_group_cache: Mutex::new(HashMap::new()),
            ibl_bind_group_cache: Mutex::new(HashMap::new()),
            dynamic_uniform_bind_group_cache: Mutex::new(Vec::new()),
            post_uniform_buffer: Mutex::new(None),
            color_texture_cache: Mutex::new(HashMap::new()),
            post_bind_group_cache: Mutex::new(HashMap::new()),
            scene_color_texture_cache: Mutex::new(HashMap::new()),
            post_texture_cache: Mutex::new(HashMap::new()),
            readback_buffer_cache: Mutex::new(HashMap::new()),
            mesh_buffer_cache: Mutex::new(HashMap::new()),
            native_mesh_buffer_cache: Mutex::new(HashMap::new()),
            state_pipeline_cache: Mutex::new(HashMap::new()),
            custom_pipeline_cache: Mutex::new(HashMap::new()),
            shadow_sampler,
            _default_texture: default_texture,
            _default_normal_map_texture: default_normal_map,
            default_texture_bind_group,
            default_normal_map_bind_group,
            default_mr_map_bind_group,
            default_emissive_map_bind_group,
            default_ibl_bind_group,
            default_ao_map_bind_group,
            _default_physical_layers_texture: default_physical_layers_texture,
            default_shadow_bind_group,
            _default_shadow_texture: default_shadow_texture,
        })
    }
}

fn storage_format_supported(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> bool {
    adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(wgpu::TextureUsages::STORAGE_BINDING)
}
