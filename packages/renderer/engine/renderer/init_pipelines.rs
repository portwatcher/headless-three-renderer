use super::*;

pub(super) struct RendererPipelines {
    pub(super) pipeline_layout: wgpu::PipelineLayout,
    pub(super) default_shadow_texture: wgpu::Texture,
    pub(super) default_shadow_bind_group: wgpu::BindGroup,
    pub(super) shadow_pipelines: [wgpu::RenderPipeline; MAX_SHADOW_LAYERS],
    pub(super) line_shadow_pipelines: [wgpu::RenderPipeline; MAX_SHADOW_LAYERS],
    pub(super) pipelines: [wgpu::RenderPipeline; 3],
    pub(super) pipelines_msaa4: [wgpu::RenderPipeline; 3],
    pub(super) transparent_pipelines: [wgpu::RenderPipeline; 3],
    pub(super) transparent_pipelines_msaa4: [wgpu::RenderPipeline; 3],
    pub(super) line_pipelines: [wgpu::RenderPipeline; 2],
    pub(super) line_pipelines_msaa4: [wgpu::RenderPipeline; 2],
    pub(super) point_pipelines: [wgpu::RenderPipeline; 2],
    pub(super) point_pipelines_msaa4: [wgpu::RenderPipeline; 2],
    pub(super) post_pipeline: wgpu::RenderPipeline,
    pub(super) background_pipeline: wgpu::RenderPipeline,
    pub(super) background_pipeline_msaa4: wgpu::RenderPipeline,
}

pub(super) fn create_renderer_pipelines(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    post_shader: &wgpu::ShaderModule,
    background_shader: &wgpu::ShaderModule,
    layouts: &RendererLayouts,
    defaults: &RendererDefaults,
) -> RendererPipelines {
    let uniform_layout = &layouts.uniform_layout;
    let texture_layout = &layouts.texture_layout;
    let normal_map_layout = &layouts.normal_map_layout;
    let mr_map_layout = &layouts.mr_map_layout;
    let emissive_map_layout = &layouts.emissive_map_layout;
    let ao_map_layout = &layouts.ao_map_layout;
    let shadow_layout = &layouts.shadow_layout;
    let ibl_layout = &layouts.ibl_layout;
    let post_layout = &layouts.post_layout;
    let background_layout = &layouts.background_layout;
    let sampler = &defaults.sampler;
    let shadow_sampler = &defaults.shadow_sampler;
    let default_texture_view = &defaults.default_texture_view;
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("headless-three-renderer pipeline layout"),
        bind_group_layouts: &[
            Some(&uniform_layout),
            Some(&texture_layout),
            Some(&normal_map_layout),
            Some(&mr_map_layout),
            Some(&emissive_map_layout),
            Some(&ibl_layout),
            Some(&ao_map_layout),
            Some(&shadow_layout),
        ],
        immediate_size: 0,
    });

    // 1x1 depth texture used as a "no shadow" default binding.
    let default_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-three-renderer default shadow texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let default_shadow_view = default_shadow_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let default_shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three-renderer default shadow bind group"),
        layout: &shadow_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&default_shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&default_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Dedicated pipeline layout for the shadow pass. The sparse group indices match the main
    // material pipeline so alpha-tested shadow casters can reuse base/alpha texture bindings.
    let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("headless-three-renderer shadow pipeline layout"),
        bind_group_layouts: &[
            Some(&uniform_layout),
            Some(&texture_layout),
            Some(&normal_map_layout),
            Some(&mr_map_layout),
            Some(&emissive_map_layout),
            Some(&ibl_layout),
            Some(&ao_map_layout),
        ],
        immediate_size: 0,
    });
    let make_shadow_pipeline =
        |entry_point: &'static str, topology: wgpu::PrimitiveTopology, label: &'static str| {
            let depth_bias = if topology == wgpu::PrimitiveTopology::TriangleList {
                // Slight slope-scaled bias to reduce acne (in addition to the
                // per-fragment bias we apply during shadow sampling).
                wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                }
            } else {
                wgpu::DepthBiasState::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Vertex::layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // No culling: captures shadows from any side, including DoubleSide materials.
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: depth_bias,
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
    let shadow_pipelines = [
        make_shadow_pipeline(
            "vs_shadow0",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 0",
        ),
        make_shadow_pipeline(
            "vs_shadow1",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 1",
        ),
        make_shadow_pipeline(
            "vs_shadow2",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 2",
        ),
        make_shadow_pipeline(
            "vs_shadow3",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 3",
        ),
        make_shadow_pipeline(
            "vs_shadow4",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 4",
        ),
        make_shadow_pipeline(
            "vs_shadow5",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 5",
        ),
        make_shadow_pipeline(
            "vs_shadow6",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 6",
        ),
        make_shadow_pipeline(
            "vs_shadow7",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 7",
        ),
        make_shadow_pipeline(
            "vs_shadow8",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 8",
        ),
        make_shadow_pipeline(
            "vs_shadow9",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 9",
        ),
        make_shadow_pipeline(
            "vs_shadow10",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 10",
        ),
        make_shadow_pipeline(
            "vs_shadow11",
            wgpu::PrimitiveTopology::TriangleList,
            "headless-three-renderer shadow pipeline 11",
        ),
    ];
    let line_shadow_pipelines = [
        make_shadow_pipeline(
            "vs_shadow0",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 0",
        ),
        make_shadow_pipeline(
            "vs_shadow1",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 1",
        ),
        make_shadow_pipeline(
            "vs_shadow2",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 2",
        ),
        make_shadow_pipeline(
            "vs_shadow3",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 3",
        ),
        make_shadow_pipeline(
            "vs_shadow4",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 4",
        ),
        make_shadow_pipeline(
            "vs_shadow5",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 5",
        ),
        make_shadow_pipeline(
            "vs_shadow6",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 6",
        ),
        make_shadow_pipeline(
            "vs_shadow7",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 7",
        ),
        make_shadow_pipeline(
            "vs_shadow8",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 8",
        ),
        make_shadow_pipeline(
            "vs_shadow9",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 9",
        ),
        make_shadow_pipeline(
            "vs_shadow10",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 10",
        ),
        make_shadow_pipeline(
            "vs_shadow11",
            wgpu::PrimitiveTopology::LineList,
            "headless-three-renderer line shadow pipeline 11",
        ),
    ];

    let vertex_buffers = [Vertex::layout()];
    let make_pipeline = |topology: Topology,
                         side: MeshSide,
                         transparent: bool,
                         sample_count: u32| {
        let label = match (topology, side, transparent) {
            (Topology::Triangles, MeshSide::Front, false) => "pipeline (tri front)",
            (Topology::Triangles, MeshSide::Back, false) => "pipeline (tri back)",
            (Topology::Triangles, MeshSide::Double, false) => "pipeline (tri double)",
            (Topology::Triangles, MeshSide::Front, true) => "pipeline (tri front, transparent)",
            (Topology::Triangles, MeshSide::Back, true) => "pipeline (tri back, transparent)",
            (Topology::Triangles, MeshSide::Double, true) => "pipeline (tri double, transparent)",
            (Topology::Lines, _, false) => "pipeline (lines)",
            (Topology::Lines, _, true) => "pipeline (lines, transparent)",
            (Topology::Points, _, false) => "pipeline (points)",
            (Topology::Points, _, true) => "pipeline (points, transparent)",
        };
        let color_targets = [Some(color_target_state(
            default_blend_state(transparent),
            true,
        ))];
        // Lines and points have no faces to cull.
        let cull_mode = match topology {
            Topology::Triangles => side.cull_mode(),
            Topology::Lines | Topology::Points => None,
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: topology.primitive(),
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: multisample_state(sample_count, false),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &color_targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    let pipelines = [
        make_pipeline(Topology::Triangles, MeshSide::Front, false, 1),
        make_pipeline(Topology::Triangles, MeshSide::Back, false, 1),
        make_pipeline(Topology::Triangles, MeshSide::Double, false, 1),
    ];
    let pipelines_msaa4 = [
        make_pipeline(Topology::Triangles, MeshSide::Front, false, 4),
        make_pipeline(Topology::Triangles, MeshSide::Back, false, 4),
        make_pipeline(Topology::Triangles, MeshSide::Double, false, 4),
    ];
    let transparent_pipelines = [
        make_pipeline(Topology::Triangles, MeshSide::Front, true, 1),
        make_pipeline(Topology::Triangles, MeshSide::Back, true, 1),
        make_pipeline(Topology::Triangles, MeshSide::Double, true, 1),
    ];
    let transparent_pipelines_msaa4 = [
        make_pipeline(Topology::Triangles, MeshSide::Front, true, 4),
        make_pipeline(Topology::Triangles, MeshSide::Back, true, 4),
        make_pipeline(Topology::Triangles, MeshSide::Double, true, 4),
    ];
    let line_pipelines = [
        make_pipeline(Topology::Lines, MeshSide::Front, false, 1),
        make_pipeline(Topology::Lines, MeshSide::Front, true, 1),
    ];
    let line_pipelines_msaa4 = [
        make_pipeline(Topology::Lines, MeshSide::Front, false, 4),
        make_pipeline(Topology::Lines, MeshSide::Front, true, 4),
    ];
    let point_pipelines = [
        make_pipeline(Topology::Points, MeshSide::Front, false, 1),
        make_pipeline(Topology::Points, MeshSide::Front, true, 1),
    ];
    let point_pipelines_msaa4 = [
        make_pipeline(Topology::Points, MeshSide::Front, false, 4),
        make_pipeline(Topology::Points, MeshSide::Front, true, 4),
    ];

    let screen_color_targets = [Some(color_target_state(None, true))];
    let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("headless-three-renderer post pipeline layout"),
        bind_group_layouts: &[Some(&post_layout)],
        immediate_size: 0,
    });
    let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("headless-three-renderer post pipeline"),
        layout: Some(&post_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &post_shader,
            entry_point: Some("vs_post"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &post_shader,
            entry_point: Some("fs_post"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &screen_color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    let background_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three-renderer background pipeline layout"),
            bind_group_layouts: &[Some(&background_layout)],
            immediate_size: 0,
        });
    let make_background_pipeline = |sample_count: u32, label: &'static str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&background_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &background_shader,
                entry_point: Some("vs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: multisample_state(sample_count, false),
            fragment: Some(wgpu::FragmentState {
                module: &background_shader,
                entry_point: Some("fs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &screen_color_targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let background_pipeline =
        make_background_pipeline(1, "headless-three-renderer background pipeline");
    let background_pipeline_msaa4 =
        make_background_pipeline(4, "headless-three-renderer background pipeline msaa4");

    RendererPipelines {
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
    }
}
