const MAX_LIGHTS: u32 = 64u;
const MAX_SHADOW_LAYERS: u32 = 12u;
const MAX_CLIPPING_PLANES: u32 = 8u;

struct GpuLight {
  light_type: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  color_intensity: vec4<f32>,
  position: vec4<f32>,
  direction: vec4<f32>,
  params: vec4<f32>,
};

struct Uniforms {
  mvp: mat4x4<f32>,
  view: mat4x4<f32>,
  model: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  camera_pos: vec4<f32>,
  base_color: vec4<f32>,
  emissive: vec4<f32>,
  metallic: f32,
  roughness: f32,
  ambient_intensity: f32,
  num_lights: u32,
  ambient_color: vec4<f32>,
  light_probe: array<vec4<f32>, 9>,
  light_probe_params: vec4<f32>,
  normal_map_params: vec4<f32>,
  ibl_params: vec4<f32>,
  env_map_params: vec4<f32>,
  ao_params: vec4<f32>,
  render_params: vec4<f32>,
  output_params: vec4<f32>,
  // texture_transform1.w = base texture uses secondary UV stream.
  // texture_transform2.w = base texture is sRGB and must be decoded to linear before shading.
  texture_transform1: vec4<f32>,
  texture_transform2: vec4<f32>,
  // alpha_map_transform1.w = alpha map is sRGB.
  // alpha_map_transform2.w = alpha map uses secondary UV stream.
  alpha_map_transform1: vec4<f32>,
  alpha_map_transform2: vec4<f32>,
  map_transform_rows: array<vec4<f32>, 12>,
  physical_map_transform_rows: array<vec4<f32>, 24>,
  clipping_planes: array<vec4<f32>, 8>,
  // x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
  clipping_params: vec4<f32>,
  fog_color: vec4<f32>,
  fog_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 12>,
  shadow_params: vec4<f32>,
  shadow_params2: vec4<f32>,
  shadow_params3: vec4<f32>,
  shadow_params4: vec4<f32>,
  shadow_infos: array<vec4<f32>, 12>,
  shadow_biases: array<vec4<f32>, 12>,
  shadow_cascade_splits: array<vec4<f32>, 12>,
  physical_params1: vec4<f32>,
  physical_params2: vec4<f32>,
  physical_params3: vec4<f32>,
  physical_params4: vec4<f32>,
  attenuation_color: vec4<f32>,
  physical_specular: vec4<f32>,
  iridescence_params: vec4<f32>,
  lights: array<GpuLight, 64>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;
@group(6) @binding(6)
var t_alpha: texture_2d<f32>;
@group(6) @binding(9)
var s_alpha: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec4<f32>,
  @location(3) color: vec4<f32>,
  @location(4) uv: vec2<f32>,
  @location(5) uv2: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) world_tangent: vec3<f32>,
  @location(3) tangent_w: f32,
  @location(4) color: vec4<f32>,
  @location(5) uv: vec2<f32>,
  @location(6) uv2: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
  output.world_pos = world_pos.xyz;
  output.world_normal = normalize((uniforms.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
  output.world_tangent = normalize((uniforms.model * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
  output.tangent_w = input.tangent.w;
  output.color = input.color;
  output.uv = input.uv;
  output.uv2 = input.uv2;
  return output;
}

fn srgb_to_linear_channel(value: f32) -> f32 {
  if value <= 0.04045 {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    srgb_to_linear_channel(color.r),
    srgb_to_linear_channel(color.g),
    srgb_to_linear_channel(color.b),
  );
}

fn decode_color_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.texture_transform2.w > 0.5 {
    return vec4<f32>(srgb_to_linear(sample.rgb), sample.a);
  }
  return sample;
}

fn decode_alpha_map_sample(sample: f32) -> f32 {
  if uniforms.alpha_map_transform1.w > 0.5 {
    return srgb_to_linear_channel(sample);
  }
  return sample;
}

fn is_clipped_by_planes(world_pos: vec3<f32>) -> bool {
  let total_count = min(u32(uniforms.clipping_params.y), MAX_CLIPPING_PLANES);
  let union_count = min(u32(uniforms.clipping_params.x), total_count);

  for (var i = 0u; i < MAX_CLIPPING_PLANES; i = i + 1u) {
    if i < union_count {
      let plane = uniforms.clipping_planes[i];
      if dot(plane.xyz, world_pos) + plane.w < 0.0 {
        return true;
      }
    }
  }

  if union_count < total_count {
    var clipped = true;
    for (var i = 0u; i < MAX_CLIPPING_PLANES; i = i + 1u) {
      if i >= union_count && i < total_count {
        let plane = uniforms.clipping_planes[i];
        clipped = clipped && (dot(plane.xyz, world_pos) + plane.w < 0.0);
      }
    }
    if clipped {
      return true;
    }
  }

  return false;
}

fn clipping_plane_alpha_coverage(world_pos: vec3<f32>) -> f32 {
  let total_count = min(u32(uniforms.clipping_params.y), MAX_CLIPPING_PLANES);
  let union_count = min(u32(uniforms.clipping_params.x), total_count);
  var clip_opacity = 1.0;

  for (var i = 0u; i < MAX_CLIPPING_PLANES; i = i + 1u) {
    if i < union_count {
      let plane = uniforms.clipping_planes[i];
      let distance_to_plane = dot(plane.xyz, world_pos) + plane.w;
      let distance_gradient = max((abs(dpdx(distance_to_plane)) + abs(dpdy(distance_to_plane))) * 0.5, 0.000001);
      clip_opacity = clip_opacity * smoothstep(-distance_gradient, distance_gradient, distance_to_plane);
    }
  }

  if union_count < total_count {
    var intersection_clip_opacity = 1.0;
    for (var i = 0u; i < MAX_CLIPPING_PLANES; i = i + 1u) {
      if i >= union_count && i < total_count {
        let plane = uniforms.clipping_planes[i];
        let distance_to_plane = dot(plane.xyz, world_pos) + plane.w;
        let distance_gradient = max((abs(dpdx(distance_to_plane)) + abs(dpdy(distance_to_plane))) * 0.5, 0.000001);
        intersection_clip_opacity = intersection_clip_opacity * (1.0 - smoothstep(-distance_gradient, distance_gradient, distance_to_plane));
      }
    }
    clip_opacity = clip_opacity * (1.0 - intersection_clip_opacity);
  }

  return clip_opacity;
}

fn alpha_test_alpha_coverage(alpha: f32, cutoff: f32) -> f32 {
  let alpha_gradient = max(abs(dpdx(alpha)) + abs(dpdy(alpha)), 0.000001);
  return smoothstep(cutoff, cutoff + alpha_gradient, alpha);
}

fn alpha_hash_threshold(position: vec4<f32>) -> f32 {
  let pixel = floor(position.xy);
  return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

fn transform_alpha_map_uv(uv: vec2<f32>) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(dot(uniforms.alpha_map_transform1.xyz, uv1), dot(uniforms.alpha_map_transform2.xyz, uv1));
}

fn output_color(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
  if uniforms.clipping_params.w > 0.5 {
    return vec4<f32>(rgb * alpha, alpha);
  }
  return vec4<f32>(rgb, alpha);
}

fn custom_fragment_body(
  input: VertexOutput,
  front_facing: bool,
  alpha: f32,
  normal: vec3<f32>,
  base_color: vec4<f32>,
) -> vec4<f32> {
  __CUSTOM_FRAGMENT_BODY__
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  var clip_opacity = 1.0;
  if uniforms.output_params.z > 0.5 {
    clip_opacity = clipping_plane_alpha_coverage(input.world_pos);
    if clip_opacity <= 0.0 {
      discard;
    }
  } else if is_clipped_by_planes(input.world_pos) {
    discard;
  }

  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let uv2 = vec2<f32>(input.uv2.x, 1.0 - input.uv2.y);
  let map_uv = select(uv, uv2, uniforms.texture_transform1.w > 0.5);
  let uv1 = vec3<f32>(map_uv, 1.0);
  let transformed_uv = vec2<f32>(dot(uniforms.texture_transform1.xyz, uv1), dot(uniforms.texture_transform2.xyz, uv1));
  let texture_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, transformed_uv));
  let base_color = texture_color * input.color * uniforms.base_color;
  var alpha = base_color.a;
  if uniforms.ao_params.z > 0.5 {
    let alpha_uv = select(uv, uv2, uniforms.alpha_map_transform2.w > 0.5);
    alpha = alpha * decode_alpha_map_sample(textureSample(t_alpha, s_alpha, transform_alpha_map_uv(alpha_uv)).g);
  }
  alpha = alpha * clip_opacity;
  let alpha_cutoff = uniforms.emissive.w;
  if alpha_cutoff > 0.0 {
    if uniforms.output_params.z > 0.5 {
      alpha = alpha_test_alpha_coverage(alpha, alpha_cutoff);
      if alpha <= 0.0 {
        discard;
      }
    } else if alpha < alpha_cutoff {
      discard;
    }
  }
  if uniforms.clipping_params.z > 0.5 && alpha < alpha_hash_threshold(input.position) {
    discard;
  }
  var normal = normalize(input.world_normal);
  if !front_facing {
    normal = -normal;
  }
  let custom_color = custom_fragment_body(input, front_facing, alpha, normal, base_color);
  return output_color(custom_color.rgb, custom_color.a);
}
