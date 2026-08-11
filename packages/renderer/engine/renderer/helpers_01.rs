use super::*;

pub(super) fn texture_mip_level_count(width: u32, height: u32, mipmap_filter: MipmapFilter) -> u32 {
    if mipmap_filter == MipmapFilter::None {
        return 1;
    }

    let mut levels = 1;
    let mut mip_width = width.max(1);
    let mut mip_height = height.max(1);
    while mip_width > 1 || mip_height > 1 {
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
        levels += 1;
    }
    levels
}

pub(super) fn texture_upload_mip_level_count(texture: &PreparedTexture) -> u32 {
    if texture.mipmaps.is_empty() {
        texture_mip_level_count(texture.width, texture.height, texture.mipmap_filter)
    } else {
        1 + texture.mipmaps.len() as u32
    }
}

pub(super) fn downsample_rgba_mip(source: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut output = vec![0u8; (next_width * next_height * 4) as usize];

    for y in 0..next_height {
        let source_y0 = y * height / next_height;
        let source_y1 = ((y + 1) * height / next_height).max(source_y0 + 1);
        for x in 0..next_width {
            let source_x0 = x * width / next_width;
            let source_x1 = ((x + 1) * width / next_width).max(source_x0 + 1);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for source_y in source_y0..source_y1.min(height) {
                for source_x in source_x0..source_x1.min(width) {
                    let source_index = ((source_y * width + source_x) * 4) as usize;
                    sum[0] += source[source_index] as u32;
                    sum[1] += source[source_index + 1] as u32;
                    sum[2] += source[source_index + 2] as u32;
                    sum[3] += source[source_index + 3] as u32;
                    count += 1;
                }
            }
            let output_index = ((y * next_width + x) * 4) as usize;
            output[output_index] = (sum[0] / count) as u8;
            output[output_index + 1] = (sum[1] / count) as u8;
            output[output_index + 2] = (sum[2] / count) as u8;
            output[output_index + 3] = (sum[3] / count) as u8;
        }
    }

    (output, next_width, next_height)
}

pub(super) fn side_index(side: MeshSide) -> usize {
    match side {
        MeshSide::Front => 0,
        MeshSide::Back => 1,
        MeshSide::Double => 2,
    }
}

pub(super) fn shadow_kind_mode(kind: ShadowKind) -> f32 {
    match kind {
        ShadowKind::DirectionalOrSpot => 0.0,
        ShadowKind::Point => 1.0,
        ShadowKind::Cascaded => 2.0,
    }
}

pub(super) fn shadow_light_space_matrices(
    settings: &RenderSettings,
) -> [[[f32; 4]; 4]; MAX_SHADOW_LAYERS] {
    let mut matrices = [Mat4::IDENTITY.to_cols_array_2d(); MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for caster in &shadow_maps.casters {
            for local_layer in 0..caster.layer_count as usize {
                let global_layer = caster.layer_base as usize + local_layer;
                matrices[global_layer] = caster.light_vps[local_layer].to_cols_array_2d();
            }
        }
    }
    matrices
}

pub(super) fn shadow_params(settings: &RenderSettings, receive_shadow: bool) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [0.0; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [0.0; 4];
    };
    [
        shadow_maps.casters.len() as f32,
        first.bias,
        first.normal_bias,
        if receive_shadow { 1.0 } else { 0.0 },
    ]
}

pub(super) fn shadow_params2(settings: &RenderSettings) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [0.0; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [0.0; 4];
    };
    [
        first.light_index as f32,
        1.0 / shadow_maps.map_width as f32,
        1.0 / shadow_maps.map_height as f32,
        settings.shadow_map_type,
    ]
}

pub(super) fn shadow_params3(settings: &RenderSettings) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [f32::MAX; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [f32::MAX; 4];
    };
    [
        first.cascade_splits[0],
        first.cascade_splits[1],
        first.cascade_splits[2],
        first.layer_count as f32,
    ]
}

pub(super) fn shadow_infos(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut infos = [[0.0; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            infos[slot] = [
                caster.light_index as f32,
                caster.layer_base as f32,
                caster.layer_count as f32,
                shadow_kind_mode(caster.kind),
            ];
        }
    }
    infos
}

pub(super) fn shadow_biases(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut biases = [[0.0; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            biases[slot] = [
                caster.bias,
                caster.normal_bias,
                caster.radius,
                caster.intensity,
            ];
        }
    }
    biases
}

pub(super) fn shadow_cascade_splits(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut splits = [[f32::MAX; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            splits[slot] = [
                caster.cascade_splits[0],
                caster.cascade_splits[1],
                caster.cascade_splits[2],
                0.0,
            ];
        }
    }
    splits
}

pub(super) fn shadow_side_mode(side: MeshSide) -> f32 {
    match side {
        MeshSide::Double => 0.0,
        MeshSide::Front => 1.0,
        MeshSide::Back => 2.0,
    }
}

pub(super) fn shadow_alpha_cutoff(mesh: &PreparedMesh) -> f32 {
    if mesh.alpha_to_coverage {
        0.5_f32.max(mesh.alpha_test)
    } else {
        0.0
    }
}

pub(super) fn apply_output_region(pass: &mut wgpu::RenderPass<'_>, settings: &RenderSettings) {
    if let Some(viewport) = settings.viewport {
        pass.set_viewport(
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            0.0,
            1.0,
        );
    }
    if let Some(scissor) = settings.scissor {
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
    }
}

pub(super) fn multisample_state(
    sample_count: u32,
    alpha_to_coverage: bool,
) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: alpha_to_coverage && sample_count > 1,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum PipelineKey {
    Tri(MeshSide),
    Line,
    Point,
}

pub(super) fn pipeline_key(mesh: &GpuMesh) -> PipelineKey {
    match mesh.topology {
        Topology::Triangles => PipelineKey::Tri(mesh.side),
        Topology::Lines => PipelineKey::Line,
        Topology::Points => PipelineKey::Point,
    }
}

pub(super) fn requires_pipeline_override(mesh: &PreparedMesh, sample_count: u32) -> bool {
    let default_blending = if mesh.is_transparent {
        BlendMode::Normal
    } else {
        BlendMode::None
    };
    !mesh.depth_test
        || mesh.depth_func != StencilCompare::LessEqual
        || !mesh.depth_write
        || !mesh.color_write
        || mesh.polygon_offset
        || mesh.stencil_write
        || (mesh.alpha_to_coverage && sample_count > 1)
        || (mesh.premultiplied_alpha
            && effective_blend_mode(mesh.blending, mesh.is_transparent) != BlendMode::None)
        || effective_blend_mode(mesh.blending, mesh.is_transparent) != default_blending
}

pub(super) fn depth_compare_function(mesh: &PreparedMesh) -> wgpu::CompareFunction {
    if mesh.depth_test {
        stencil_compare(mesh.depth_func)
    } else {
        wgpu::CompareFunction::Always
    }
}

pub(super) fn depth_bias_state(mesh: &PreparedMesh) -> wgpu::DepthBiasState {
    if mesh.polygon_offset {
        wgpu::DepthBiasState {
            constant: mesh.polygon_offset_units,
            slope_scale: mesh.polygon_offset_factor,
            clamp: 0.0,
        }
    } else {
        wgpu::DepthBiasState::default()
    }
}

pub(super) fn stencil_state(mesh: &PreparedMesh) -> wgpu::StencilState {
    if !mesh.stencil_write {
        return wgpu::StencilState::default();
    }
    let face = wgpu::StencilFaceState {
        compare: stencil_compare(mesh.stencil_func),
        fail_op: stencil_operation(mesh.stencil_fail),
        depth_fail_op: stencil_operation(mesh.stencil_z_fail),
        pass_op: stencil_operation(mesh.stencil_z_pass),
    };
    wgpu::StencilState {
        front: face,
        back: face,
        read_mask: mesh.stencil_func_mask,
        write_mask: mesh.stencil_write_mask,
    }
}

pub(super) fn stencil_compare(compare: StencilCompare) -> wgpu::CompareFunction {
    match compare {
        StencilCompare::Never => wgpu::CompareFunction::Never,
        StencilCompare::Less => wgpu::CompareFunction::Less,
        StencilCompare::Equal => wgpu::CompareFunction::Equal,
        StencilCompare::LessEqual => wgpu::CompareFunction::LessEqual,
        StencilCompare::Greater => wgpu::CompareFunction::Greater,
        StencilCompare::NotEqual => wgpu::CompareFunction::NotEqual,
        StencilCompare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        StencilCompare::Always => wgpu::CompareFunction::Always,
    }
}

pub(super) fn stencil_operation(operation: StencilOperation) -> wgpu::StencilOperation {
    match operation {
        StencilOperation::Zero => wgpu::StencilOperation::Zero,
        StencilOperation::Keep => wgpu::StencilOperation::Keep,
        StencilOperation::Replace => wgpu::StencilOperation::Replace,
        StencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        StencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
        StencilOperation::Invert => wgpu::StencilOperation::Invert,
    }
}

pub(super) fn color_target_state(
    blend: Option<wgpu::BlendState>,
    color_write: bool,
) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: COLOR_FORMAT,
        blend,
        write_mask: if color_write {
            wgpu::ColorWrites::ALL
        } else {
            wgpu::ColorWrites::empty()
        },
    }
}

pub(super) fn default_blend_state(transparent: bool) -> Option<wgpu::BlendState> {
    if transparent {
        blend_state(BlendMode::Normal, None, true, false)
    } else {
        None
    }
}

pub(super) fn effective_blend_mode(mode: BlendMode, is_transparent: bool) -> BlendMode {
    match mode {
        BlendMode::Normal if !is_transparent => BlendMode::None,
        other => other,
    }
}

pub(super) fn blend_state(
    mode: BlendMode,
    custom: Option<CustomBlendState>,
    is_transparent: bool,
    premultiplied_alpha: bool,
) -> Option<wgpu::BlendState> {
    match effective_blend_mode(mode, is_transparent) {
        BlendMode::None => None,
        BlendMode::Normal => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: if premultiplied_alpha {
                    wgpu::BlendFactor::One
                } else {
                    wgpu::BlendFactor::SrcAlpha
                },
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: if premultiplied_alpha {
                    wgpu::BlendFactor::One
                } else {
                    wgpu::BlendFactor::SrcAlpha
                },
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Subtractive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Multiply => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Custom => custom.map(custom_blend_state),
    }
}

pub(super) fn custom_blend_state(state: CustomBlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: blend_factor(state.color_src_factor),
            dst_factor: blend_factor(state.color_dst_factor),
            operation: blend_operation(state.color_equation),
        },
        alpha: wgpu::BlendComponent {
            src_factor: blend_factor(state.alpha_src_factor),
            dst_factor: blend_factor(state.alpha_dst_factor),
            operation: blend_operation(state.alpha_equation),
        },
    }
}

pub(super) fn blend_operation(equation: BlendEquation) -> wgpu::BlendOperation {
    match equation {
        BlendEquation::Add => wgpu::BlendOperation::Add,
        BlendEquation::Subtract => wgpu::BlendOperation::Subtract,
        BlendEquation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendEquation::Min => wgpu::BlendOperation::Min,
        BlendEquation::Max => wgpu::BlendOperation::Max,
    }
}

pub(super) fn blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
    match factor {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SrcColor => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::DstColor => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::SrcAlphaSaturate => wgpu::BlendFactor::SrcAlphaSaturated,
        BlendFactor::ConstantColor | BlendFactor::ConstantAlpha => wgpu::BlendFactor::Constant,
        BlendFactor::OneMinusConstantColor | BlendFactor::OneMinusConstantAlpha => {
            wgpu::BlendFactor::OneMinusConstant
        }
    }
}

pub(super) fn blend_constant(state: Option<CustomBlendState>) -> Option<wgpu::Color> {
    let state = state?;
    if !uses_constant_factor(state) {
        return None;
    }
    let alpha_as_rgb = matches!(
        state.color_src_factor,
        BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
    ) || matches!(
        state.color_dst_factor,
        BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
    );
    let (r, g, b) = if alpha_as_rgb {
        (
            state.constant[3] as f64,
            state.constant[3] as f64,
            state.constant[3] as f64,
        )
    } else {
        (
            state.constant[0] as f64,
            state.constant[1] as f64,
            state.constant[2] as f64,
        )
    };
    Some(wgpu::Color {
        r,
        g,
        b,
        a: state.constant[3] as f64,
    })
}

pub(super) fn uses_constant_factor(state: CustomBlendState) -> bool {
    [
        state.color_src_factor,
        state.color_dst_factor,
        state.alpha_src_factor,
        state.alpha_dst_factor,
    ]
    .iter()
    .any(|factor| {
        matches!(
            factor,
            BlendFactor::ConstantColor
                | BlendFactor::OneMinusConstantColor
                | BlendFactor::ConstantAlpha
                | BlendFactor::OneMinusConstantAlpha
        )
    })
}
