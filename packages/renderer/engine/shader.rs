pub const SHADER: &str = r#"
const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 16u;
const MAX_CLIPPING_PLANES: u32 = 8u;

struct GpuLight {
  light_type: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  color_intensity: vec4<f32>,
  // xyz = position (point/spot) or ground_color (hemisphere), w = distance
  position: vec4<f32>,
  // xyz = direction, w = decay
  direction: vec4<f32>,
  // spot: x = cos(outer_angle), y = cos(inner_angle); rect area: x = width, y = height
  params: vec4<f32>,
};

struct Uniforms {
  mvp: mat4x4<f32>,
  view: mat4x4<f32>,
  model: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  camera_pos: vec4<f32>,
  base_color: vec4<f32>,
  emissive: vec4<f32>,  // xyz = emissive color, w = alpha test cutoff
  metallic: f32,
  roughness: f32,
  ambient_intensity: f32,
  num_lights: u32,
  ambient_color: vec4<f32>,
  // xyz = LightProbe SH coefficient, w = reserved.
  light_probe: array<vec4<f32>, 9>,
  // x = has LightProbe, y = has toon gradient map, z = depth packing, w = has matcap color map.
  light_probe_params: vec4<f32>,
  // x/y = normalScale or bumpScale, z = normal mode (0=none, 1=normalMap, 2=bumpMap), w = has_ibl
  normal_map_params: vec4<f32>,
  // x = env_intensity, y = shading_model (0=standard PBR, 1=basic/unlit, 2=lambert, 3=normal, 4=matcap, 5=phong, 6=depth, 7=toon, 8=distance, 9=shadow), z = camera near, w = camera far
  ibl_params: vec4<f32>,
  // x = ao_map_intensity, y = has_ao_map, z = has_alpha_map, w = has_light_map
  ao_params: vec4<f32>,
  // x = 1/width, y = 1/height, z = width, w = height
  render_params: vec4<f32>,
  // x = 1 for LinearSRGBColorSpace output, 0 for SRGBColorSpace output.
  output_params: vec4<f32>,
  // texture_transform1.xyz / texture_transform2.xyz = base-color texture transform rows.
  // texture_transform1.w = base texture uses secondary UV stream.
  // texture_transform2.w = base texture is sRGB and must be decoded to linear before shading.
  texture_transform1: vec4<f32>,
  texture_transform2: vec4<f32>,
  // alpha_map_transform1.xyz / alpha_map_transform2.xyz = alpha-map texture transform rows.
  // alpha_map_transform2.w = alpha map uses secondary UV stream.
  alpha_map_transform1: vec4<f32>,
  alpha_map_transform2: vec4<f32>,
  // Row pairs for normal, metallic-roughness, emissive, AO, light, and specular map transforms.
  // map_transform_rows[1].w = active normal/bump map uses secondary UV stream.
  // map_transform_rows[3].w = metallic-roughness map uses secondary UV stream.
  // map_transform_rows[4].w = emissive map is sRGB.
  // map_transform_rows[5].w = emissive map uses secondary UV stream.
  // map_transform_rows[8].w = light map is sRGB.
  map_transform_rows: array<vec4<f32>, 12>,
  // Row pairs for clearcoat, clearcoat roughness, clearcoat normal, sheen color,
  // sheen roughness, anisotropy, transmission, and thickness map transforms.
  // physical_map_transform_rows[1].w = clearcoat map uses secondary UV stream.
  // physical_map_transform_rows[3].w = clearcoat roughness map uses secondary UV stream.
  // physical_map_transform_rows[5].w = clearcoat normal map uses secondary UV stream.
  // physical_map_transform_rows[7].w = sheen color map uses secondary UV stream, or matcap color map when shading_model is matcap.
  // physical_map_transform_rows[9].w = sheen roughness map uses secondary UV stream.
  // physical_map_transform_rows[11].w = anisotropy map uses secondary UV stream.
  // physical_map_transform_rows[13].w = transmission map uses secondary UV stream.
  // physical_map_transform_rows[15].w = thickness map uses secondary UV stream.
  // physical_map_transform_rows[17].w = specular color map uses secondary UV stream.
  // physical_map_transform_rows[19].w = specular intensity map uses secondary UV stream.
  physical_map_transform_rows: array<vec4<f32>, 20>,
  // World-space clipping planes [normal.xyz, constant].
  clipping_planes: array<vec4<f32>, 8>,
  // x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
  clipping_params: vec4<f32>,
  // xyz = fog color
  fog_color: vec4<f32>,
  // x = mode (0=off, 1=linear, 2=exp2), y = near, z = far, w = density
  fog_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 6>,
  // x = has_shadow, y = bias, z = normal_bias, w = receive_shadow
  shadow_params: vec4<f32>,
  // x = shadow light index (as f32), y = 1/map_size, z = shadow kind (0=2D, 1=point, 2=cascaded), w = layer count
  shadow_params2: vec4<f32>,
  // xyz/w = cascade split distances; missing splits are large sentinel values
  shadow_params3: vec4<f32>,
  // x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
  physical_params1: vec4<f32>,
  // xyz = sheen color, w = sheen roughness
  physical_params2: vec4<f32>,
  // x = anisotropy, y = anisotropy rotation, z/w = thickness/attenuation distance or distance near/far
  physical_params3: vec4<f32>,
  // x/y = clearcoat normal scale, z = light_map_intensity, w = has_specular_map or matcap map sRGB flag
  physical_params4: vec4<f32>,
  // xyz = attenuation color or distance reference position
  attenuation_color: vec4<f32>,
  // xyz = MeshPhysicalMaterial specular color factor, w = specular intensity.
  physical_specular: vec4<f32>,
  lights: array<GpuLight, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@group(2) @binding(0)
var t_normal: texture_2d<f32>;
@group(2) @binding(1)
var s_normal: sampler;

@group(3) @binding(0)
var t_metallic_roughness: texture_2d<f32>;
@group(3) @binding(1)
var s_metallic_roughness: sampler;

@group(4) @binding(0)
var t_emissive: texture_2d<f32>;
@group(4) @binding(1)
var s_emissive: sampler;

@group(5) @binding(0)
var t_irradiance: texture_cube<f32>;
@group(5) @binding(1)
var t_prefilter: texture_cube<f32>;
@group(5) @binding(2)
var t_brdf_lut: texture_2d<f32>;
@group(5) @binding(3)
var s_ibl: sampler;

@group(6) @binding(0)
var t_ao: texture_2d<f32>;
@group(6) @binding(1)
var t_physical_layers: texture_2d_array<f32>;
@group(6) @binding(2)
var t_physical_sheen: texture_2d<f32>;
@group(6) @binding(3)
var t_physical_specular: texture_2d<f32>;
@group(6) @binding(4)
var t_clearcoat_normal: texture_2d<f32>;
@group(6) @binding(5)
var s_physical: sampler;
@group(6) @binding(6)
var t_alpha: texture_2d<f32>;
@group(6) @binding(7)
var t_light_map: texture_2d<f32>;
@group(6) @binding(8)
var s_ao: sampler;
@group(6) @binding(9)
var s_alpha: sampler;
@group(6) @binding(10)
var s_light_map: sampler;
@group(6) @binding(11)
var s_specular_map: sampler;
@group(6) @binding(12)
var s_physical_layers_map: sampler;
@group(6) @binding(13)
var s_physical_sheen_map: sampler;
@group(6) @binding(14)
var s_physical_specular_map: sampler;
@group(6) @binding(15)
var s_clearcoat_normal_map: sampler;

@group(7) @binding(0)
var t_shadow: texture_depth_2d_array;
@group(7) @binding(1)
var s_shadow: sampler_comparison;
@group(7) @binding(2)
var t_scene_color: texture_2d<f32>;
@group(7) @binding(3)
var s_scene_color: sampler;

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

fn shadow_position(input: VertexInput, layer: u32) -> vec4<f32> {
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  return uniforms.light_space_matrices[layer] * world_pos;
}

@vertex
fn vs_shadow0(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 0u);
}

@vertex
fn vs_shadow1(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 1u);
}

@vertex
fn vs_shadow2(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 2u);
}

@vertex
fn vs_shadow3(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 3u);
}

@vertex
fn vs_shadow4(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 4u);
}

@vertex
fn vs_shadow5(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 5u);
}

fn sample_shadow_layer(world_pos: vec3<f32>, layer: u32, world_normal: vec3<f32>) -> f32 {
  let biased_pos = world_pos + world_normal * uniforms.shadow_params.z;
  let light_ndc = uniforms.light_space_matrices[layer] * vec4<f32>(biased_pos, 1.0);
  let proj = light_ndc.xyz / light_ndc.w;

  if proj.z > 1.0 || proj.z < 0.0 {
    return 1.0;
  }
  let uv = vec2<f32>(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
  if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
    return 1.0;
  }

  let reference = proj.z - uniforms.shadow_params.y;
  let texel = uniforms.shadow_params2.y;

  // 3x3 PCF.
  var sum: f32 = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
      sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, uv + offset, layer, reference);
    }
  }
  return sum / 9.0;
}

fn point_shadow_layer(light_vec: vec3<f32>) -> u32 {
  let abs_vec = abs(light_vec);
  if abs_vec.x >= abs_vec.y && abs_vec.x >= abs_vec.z {
    return select(1u, 0u, light_vec.x >= 0.0);
  }
  if abs_vec.y >= abs_vec.x && abs_vec.y >= abs_vec.z {
    return select(3u, 2u, light_vec.y >= 0.0);
  }
  return select(5u, 4u, light_vec.z >= 0.0);
}

// 3x3 PCF shadow sampling. Returns the fraction of samples NOT in shadow
// (i.e. 1.0 = fully lit, 0.0 = fully occluded).
fn sample_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  if uniforms.shadow_params.x < 0.5 || uniforms.shadow_params.w < 0.5 {
    return 1.0;
  }
  if uniforms.shadow_params2.z > 0.5 {
    if uniforms.shadow_params2.z > 1.5 {
      let camera_dist = distance(world_pos, uniforms.camera_pos.xyz);
      var layer = 0u;
      if camera_dist > uniforms.shadow_params3.x {
        layer = 1u;
      }
      if camera_dist > uniforms.shadow_params3.y {
        layer = 2u;
      }
      if camera_dist > uniforms.shadow_params3.z {
        layer = 3u;
      }
      layer = min(layer, max(u32(uniforms.shadow_params2.w), 1u) - 1u);
      return sample_shadow_layer(world_pos, layer, world_normal);
    }
    let light_index = u32(uniforms.shadow_params2.x);
    let light_pos = uniforms.lights[light_index].position.xyz;
    let layer = point_shadow_layer(world_pos - light_pos);
    return sample_shadow_layer(world_pos, layer, world_normal);
  }
  return sample_shadow_layer(world_pos, 0u, world_normal);
}

// GGX/Trowbridge-Reitz normal distribution
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 0.0001);
}

// Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
  return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn geometry_smith_correlated_anisotropic(
  alpha_t: f32,
  alpha_b: f32,
  dot_tv: f32,
  dot_bv: f32,
  dot_tl: f32,
  dot_bl: f32,
  n_dot_v: f32,
  n_dot_l: f32,
) -> f32 {
  let gv = n_dot_l * length(vec3<f32>(alpha_t * dot_tv, alpha_b * dot_bv, n_dot_v));
  let gl = n_dot_v * length(vec3<f32>(alpha_t * dot_tl, alpha_b * dot_bl, n_dot_l));
  return 0.5 / max(gv + gl, 0.0001);
}

fn distribution_ggx_anisotropic(
  alpha_t: f32,
  alpha_b: f32,
  n_dot_h: f32,
  dot_th: f32,
  dot_bh: f32,
) -> f32 {
  let a2 = alpha_t * alpha_b;
  let v = vec3<f32>(alpha_b * dot_th, alpha_t * dot_bh, a2 * n_dot_h);
  let v2 = max(dot(v, v), 0.0001);
  let w2 = a2 / v2;
  return (a2 * w2 * w2) / PI;
}

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_f90(cos_theta: f32, f0: vec3<f32>, f90: f32) -> vec3<f32> {
  return f0 + (vec3<f32>(f90) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Schlick Fresnel with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
  return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness_f90(cos_theta: f32, f0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
  return f0 + (max(vec3<f32>(f90 * (1.0 - roughness)), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Estevez/Kulla Charlie distribution and Neubelt visibility for cloth sheen.
fn distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
  let alpha = max(roughness * roughness, 0.0001);
  let inv_alpha = 1.0 / alpha;
  let cos2h = n_dot_h * n_dot_h;
  let sin2h = max(1.0 - cos2h, 0.0078125);
  return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

fn visibility_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
  return saturate(1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 0.0001));
}

fn brdf_sheen(
  L: vec3<f32>,
  V: vec3<f32>,
  N: vec3<f32>,
  sheen_color: vec3<f32>,
  sheen_roughness: f32,
) -> vec3<f32> {
  let H = normalize(V + L);
  let n_dot_l = max(dot(N, L), 0.0);
  let n_dot_v = max(dot(N, V), 0.0);
  let n_dot_h = max(dot(N, H), 0.0);
  let D = distribution_charlie(sheen_roughness, n_dot_h);
  let Vis = visibility_neubelt(n_dot_v, n_dot_l);
  return sheen_color * (D * Vis);
}

fn ibl_sheen_brdf(N: vec3<f32>, V: vec3<f32>, roughness: f32) -> f32 {
  let n_dot_v = max(dot(N, V), 0.0);
  let r2 = roughness * roughness;
  let r_inv = 1.0 / (roughness + 0.1);
  let a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * r_inv;
  let b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * r_inv;
  return saturate(exp(a * n_dot_v + b));
}

fn tangent_basis(N: vec3<f32>, tangent: vec3<f32>, tangent_w: f32) -> mat3x3<f32> {
  var T = tangent;
  if dot(T, T) < 0.0001 {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.y) > 0.999);
    T = normalize(cross(up, N));
  } else {
    T = normalize(T - N * dot(N, T));
  }
  let B = normalize(cross(N, T) * select(1.0, tangent_w, abs(tangent_w) > 0.0));
  return mat3x3<f32>(T, B, N);
}

fn volume_attenuation(distance: f32, attenuation_color: vec3<f32>, attenuation_distance: f32) -> vec3<f32> {
  if attenuation_distance >= 1.0e19 {
    return vec3<f32>(1.0);
  }
  let safe_color = max(attenuation_color, vec3<f32>(0.0001));
  let coefficient = -log(safe_color) / max(attenuation_distance, 0.0001);
  return exp(-coefficient * distance);
}

// Three.js-compatible distance attenuation
fn get_distance_attenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
  var falloff = 1.0 / max(pow(light_distance, decay_exponent), 0.01);
  if cutoff_distance > 0.0 {
    let ratio = light_distance / cutoff_distance;
    let ratio2 = ratio * ratio;
    let ratio4 = ratio2 * ratio2;
    let window = saturate(1.0 - ratio4);
    falloff *= window * window;
  }
  return falloff;
}

// Three.js-compatible spot attenuation
fn get_spot_attenuation(cone_cos: f32, penumbra_cos: f32, angle_cos: f32) -> f32 {
  return smoothstep(cone_cos, penumbra_cos, angle_cos);
}

fn apply_fog(color: vec3<f32>, fog_distance: f32) -> vec3<f32> {
  if uniforms.fog_params.x == 1.0 {
    let fog_factor = smoothstep(uniforms.fog_params.y, uniforms.fog_params.z, fog_distance);
    return mix(color, uniforms.fog_color.rgb, fog_factor);
  }
  if uniforms.fog_params.x == 2.0 {
    let fog_density_distance = uniforms.fog_params.w * fog_distance;
    let fog_factor = clamp(1.0 - exp2(-fog_density_distance * fog_density_distance * 1.442695), 0.0, 1.0);
    return mix(color, uniforms.fog_color.rgb, fog_factor);
  }
  return color;
}

fn transform_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let map_uv = select(uv, uv2, uniforms.texture_transform1.w > 0.5);
  let uv1 = vec3<f32>(map_uv, 1.0);
  return vec2<f32>(dot(uniforms.texture_transform1.xyz, uv1), dot(uniforms.texture_transform2.xyz, uv1));
}

fn transform_alpha_map_uv(uv: vec2<f32>) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(dot(uniforms.alpha_map_transform1.xyz, uv1), dot(uniforms.alpha_map_transform2.xyz, uv1));
}

fn transform_slot_uv(uv: vec2<f32>, row_index: u32) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(
    dot(uniforms.map_transform_rows[row_index].xyz, uv1),
    dot(uniforms.map_transform_rows[row_index + 1u].xyz, uv1),
  );
}

fn transform_normal_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let normal_uv = select(uv, uv2, uniforms.map_transform_rows[1u].w > 0.5);
  return transform_slot_uv(normal_uv, 0u);
}

fn transform_metallic_roughness_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let mr_uv = select(uv, uv2, uniforms.map_transform_rows[3u].w > 0.5);
  return transform_slot_uv(mr_uv, 2u);
}

fn transform_emissive_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let emissive_uv = select(uv, uv2, uniforms.map_transform_rows[5u].w > 0.5);
  return transform_slot_uv(emissive_uv, 4u);
}

fn transform_ao_map_uv(uv: vec2<f32>) -> vec2<f32> {
  return transform_slot_uv(uv, 6u);
}

fn transform_light_map_uv(uv: vec2<f32>) -> vec2<f32> {
  return transform_slot_uv(uv, 8u);
}

fn transform_specular_map_uv(uv: vec2<f32>) -> vec2<f32> {
  return transform_slot_uv(uv, 10u);
}

fn transform_physical_slot_uv(uv: vec2<f32>, row_index: u32) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(
    dot(uniforms.physical_map_transform_rows[row_index].xyz, uv1),
    dot(uniforms.physical_map_transform_rows[row_index + 1u].xyz, uv1),
  );
}

fn transform_clearcoat_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let clearcoat_uv = select(uv, uv2, uniforms.physical_map_transform_rows[1u].w > 0.5);
  return transform_physical_slot_uv(clearcoat_uv, 0u);
}

fn transform_clearcoat_roughness_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let roughness_uv = select(uv, uv2, uniforms.physical_map_transform_rows[3u].w > 0.5);
  return transform_physical_slot_uv(roughness_uv, 2u);
}

fn transform_clearcoat_normal_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let normal_uv = select(uv, uv2, uniforms.physical_map_transform_rows[5u].w > 0.5);
  return transform_physical_slot_uv(normal_uv, 4u);
}

fn transform_sheen_color_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let sheen_uv = select(uv, uv2, uniforms.physical_map_transform_rows[7u].w > 0.5);
  return transform_physical_slot_uv(sheen_uv, 6u);
}

fn transform_matcap_color_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let map_uv = select(uv, uv2, uniforms.physical_map_transform_rows[7u].w > 0.5);
  return transform_physical_slot_uv(map_uv, 6u);
}

fn transform_sheen_roughness_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let roughness_uv = select(uv, uv2, uniforms.physical_map_transform_rows[9u].w > 0.5);
  return transform_physical_slot_uv(roughness_uv, 8u);
}

fn transform_anisotropy_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let anisotropy_uv = select(uv, uv2, uniforms.physical_map_transform_rows[11u].w > 0.5);
  return transform_physical_slot_uv(anisotropy_uv, 10u);
}

fn transform_transmission_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let transmission_uv = select(uv, uv2, uniforms.physical_map_transform_rows[13u].w > 0.5);
  return transform_physical_slot_uv(transmission_uv, 12u);
}

fn transform_thickness_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let thickness_uv = select(uv, uv2, uniforms.physical_map_transform_rows[15u].w > 0.5);
  return transform_physical_slot_uv(thickness_uv, 14u);
}

fn transform_specular_color_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let specular_uv = select(uv, uv2, uniforms.physical_map_transform_rows[17u].w > 0.5);
  return transform_physical_slot_uv(specular_uv, 16u);
}

fn transform_specular_intensity_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let specular_uv = select(uv, uv2, uniforms.physical_map_transform_rows[19u].w > 0.5);
  return transform_physical_slot_uv(specular_uv, 18u);
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

fn decode_matcap_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.physical_params4.w > 0.5 {
    return vec4<f32>(srgb_to_linear(sample.rgb), sample.a);
  }
  return sample;
}

fn decode_emissive_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.map_transform_rows[4u].w > 0.5 {
    return vec4<f32>(srgb_to_linear(sample.rgb), sample.a);
  }
  return sample;
}

fn decode_light_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.map_transform_rows[8u].w > 0.5 {
    return vec4<f32>(srgb_to_linear(sample.rgb), sample.a);
  }
  return sample;
}

fn light_probe_irradiance(normal: vec3<f32>) -> vec3<f32> {
  let x = normal.x;
  let y = normal.y;
  let z = normal.z;
  var result = uniforms.light_probe[0].rgb * 0.886227;
  result = result + uniforms.light_probe[1].rgb * (2.0 * 0.511664 * y);
  result = result + uniforms.light_probe[2].rgb * (2.0 * 0.511664 * z);
  result = result + uniforms.light_probe[3].rgb * (2.0 * 0.511664 * x);
  result = result + uniforms.light_probe[4].rgb * (2.0 * 0.429043 * x * y);
  result = result + uniforms.light_probe[5].rgb * (2.0 * 0.429043 * y * z);
  result = result + uniforms.light_probe[6].rgb * (0.743125 * z * z - 0.247708);
  result = result + uniforms.light_probe[7].rgb * (2.0 * 0.429043 * x * z);
  result = result + uniforms.light_probe[8].rgb * (0.429043 * (x * x - y * y));
  return max(result, vec3<f32>(0.0));
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

fn alpha_hash_threshold(position: vec4<f32>) -> f32 {
  let pixel = floor(position.xy);
  return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

fn bump_height(bump_uv: vec2<f32>) -> f32 {
  return uniforms.normal_map_params.x * textureSample(t_normal, s_normal, bump_uv).r;
}

fn perturb_normal_from_bump(surf_pos: vec3<f32>, surf_norm: vec3<f32>, bump_uv: vec2<f32>) -> vec3<f32> {
  let d_st_dx = dpdx(bump_uv);
  let d_st_dy = dpdy(bump_uv);
  let h_ll = bump_height(bump_uv);
  let d_h_dx = bump_height(bump_uv + d_st_dx) - h_ll;
  let d_h_dy = bump_height(bump_uv + d_st_dy) - h_ll;

  let sigma_x = normalize(dpdx(surf_pos));
  let sigma_y = normalize(dpdy(surf_pos));
  let r1 = cross(sigma_y, surf_norm);
  let r2 = cross(surf_norm, sigma_x);
  let det = dot(sigma_x, r1);
  let grad = sign(det) * (d_h_dx * r1 + d_h_dy * r2);
  return normalize(abs(det) * surf_norm - grad);
}

fn output_color(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
  if uniforms.clipping_params.w > 0.5 {
    return vec4<f32>(rgb * alpha, alpha);
  }
  return vec4<f32>(rgb, alpha);
}

fn pack_depth_to_rgba(v: f32) -> vec4<f32> {
  if v <= 0.0 {
    return vec4<f32>(0.0);
  }
  if v >= 1.0 {
    return vec4<f32>(1.0);
  }
  var vuf = floor(v * 16777216.0);
  let af = fract(v * 16777216.0);
  let bf = fract(vuf / 256.0);
  vuf = floor(vuf / 256.0);
  let gf = fract(vuf / 256.0);
  vuf = floor(vuf / 256.0);
  return vec4<f32>(vuf / 255.0, gf * (256.0 / 255.0), bf * (256.0 / 255.0), af);
}

fn pack_depth_to_rgb(v: f32) -> vec3<f32> {
  if v <= 0.0 {
    return vec3<f32>(0.0);
  }
  if v >= 1.0 {
    return vec3<f32>(1.0);
  }
  var vuf = floor(v * 65536.0);
  let bf = fract(v * 65536.0);
  let gf = fract(vuf / 256.0);
  vuf = floor(vuf / 256.0);
  return vec3<f32>(vuf / 255.0, gf * (256.0 / 255.0), bf);
}

fn pack_depth_to_rg(v: f32) -> vec2<f32> {
  if v <= 0.0 {
    return vec2<f32>(0.0);
  }
  if v >= 1.0 {
    return vec2<f32>(1.0);
  }
  let vuf = floor(v * 256.0);
  let gf = fract(v * 256.0);
  return vec2<f32>(vuf / 255.0, gf);
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  if is_clipped_by_planes(input.world_pos) {
    discard;
  }

  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let uv2 = vec2<f32>(input.uv2.x, 1.0 - input.uv2.y);
  let screen_uv = clamp(input.position.xy * uniforms.render_params.xy, vec2<f32>(0.0), vec2<f32>(1.0));
  let tex_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, transform_map_uv(uv, uv2)));
  let albedo = tex_color.rgb * input.color.rgb * uniforms.base_color.rgb;
  var alpha = tex_color.a * input.color.a * uniforms.base_color.a;
  if uniforms.ao_params.z > 0.5 {
    let alpha_uv = select(uv, uv2, uniforms.alpha_map_transform2.w > 0.5);
    alpha = alpha * textureSample(t_alpha, s_alpha, transform_alpha_map_uv(alpha_uv)).g;
  }

  // Alpha test: discard fragments below the cutoff threshold
  let alpha_cutoff = uniforms.emissive.w;
  if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
    discard;
  }
  if uniforms.clipping_params.z > 0.5 && alpha < alpha_hash_threshold(input.position) {
    discard;
  }

  let shading_model = u32(uniforms.ibl_params.y);

  // Normal mapping via TBN matrix.
  var N = normalize(input.world_normal);
  // Flip normal when shading back-facing fragments (BackSide / DoubleSide).
  // For FrontSide meshes, back faces are culled so front_facing is always true.
  if !front_facing {
    N = -N;
  }
  var tbn = tangent_basis(N, input.world_tangent, input.tangent_w);
  let normal_mode = u32(uniforms.normal_map_params.z + 0.5);
  if normal_mode == 1u {
    let normal_sample = textureSample(t_normal, s_normal, transform_normal_map_uv(uv, uv2)).rgb;
    var tangent_normal = normal_sample * 2.0 - vec3<f32>(1.0);
    tangent_normal.x *= uniforms.normal_map_params.x;
    tangent_normal.y *= uniforms.normal_map_params.y;
    N = normalize(tbn * tangent_normal);
    tbn = tangent_basis(N, tbn[0], input.tangent_w);
  } else if normal_mode == 2u {
    N = perturb_normal_from_bump(input.world_pos, N, transform_normal_map_uv(uv, uv2));
    tbn = tangent_basis(N, tbn[0], input.tangent_w);
  }

  // Ambient occlusion: sample red channel, blend toward 1.0 by intensity.
  // Matches three.js: ao = (texture.r - 1.0) * aoMapIntensity + 1.0
  var ao: f32 = 1.0;
  if uniforms.ao_params.y > 0.5 {
    let ao_sample = textureSample(t_ao, s_ao, transform_ao_map_uv(uv2)).r;
    ao = (ao_sample - 1.0) * uniforms.ao_params.x + 1.0;
  }
  let has_light_map = uniforms.ao_params.w > 0.5;
  var light_map_irradiance = vec3<f32>(0.0);
  if has_light_map {
    light_map_irradiance = decode_light_map_sample(textureSample(t_light_map, s_light_map, transform_light_map_uv(uv2))).rgb * max(uniforms.physical_params4.z, 0.0);
  }
  let light_map_diffuse = albedo * light_map_irradiance * (1.0 / PI) * ao;

  // MeshBasicMaterial: unlit. Output = albedo * ao, then emissive + tone map + gamma.
  if shading_model == 1u {
    var unlit = albedo * ao;
    if has_light_map {
      unlit = light_map_diffuse;
    }
    let emissive_basic = decode_emissive_map_sample(textureSample(t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2))).rgb;
    unlit = unlit + uniforms.emissive.rgb * emissive_basic;
    let mapped_basic = apply_output_color_space(aces_filmic_tone_mapping(unlit));
    let fogged_basic = apply_fog(mapped_basic, distance(input.world_pos, uniforms.camera_pos.xyz));
    return output_color(fogged_basic, alpha);
  }

  if shading_model == 3u {
    let view_normal = normalize((uniforms.view * vec4<f32>(N, 0.0)).xyz);
    return output_color(view_normal * 0.5 + vec3<f32>(0.5), alpha);
  }

  if shading_model == 4u {
    let view_normal = normalize((uniforms.view * vec4<f32>(N, 0.0)).xyz);
    let view_position = (uniforms.view * vec4<f32>(input.world_pos, 1.0)).xyz;
    let view_dir = normalize(-view_position);
    let matcap_x = normalize(vec3<f32>(view_dir.z, 0.0, -view_dir.x));
    let matcap_y = cross(view_dir, matcap_x);
    let matcap_uv = vec2<f32>(dot(matcap_x, view_normal), dot(matcap_y, view_normal)) * 0.495 + vec2<f32>(0.5);
    var matcap_surface_color = input.color.rgb * uniforms.base_color.rgb;
    if uniforms.light_probe_params.w > 0.5 {
      let matcap_map = decode_matcap_map_sample(textureSample(t_physical_sheen, s_physical, transform_matcap_color_map_uv(uv, uv2)));
      matcap_surface_color *= matcap_map.rgb;
    }
    var matcap_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, matcap_uv)).rgb * matcap_surface_color;
    let mapped_matcap = apply_output_color_space(aces_filmic_tone_mapping(matcap_color));
    let fogged_matcap = apply_fog(mapped_matcap, distance(input.world_pos, uniforms.camera_pos.xyz));
    return output_color(fogged_matcap, alpha);
  }

  if shading_model == 6u {
    let frag_depth = clamp(input.position.z, 0.0, 1.0);
    let depth_packing = u32(uniforms.light_probe_params.z + 0.5);
    if depth_packing == 1u {
      return pack_depth_to_rgba(frag_depth);
    }
    if depth_packing == 2u {
      return vec4<f32>(pack_depth_to_rgb(frag_depth), 1.0);
    }
    if depth_packing == 3u {
      return vec4<f32>(pack_depth_to_rg(frag_depth), 0.0, 1.0);
    }
    let depth = 1.0 - frag_depth;
    return output_color(vec3<f32>(depth), alpha);
  }

  if shading_model == 8u {
    let distance_depth = clamp(
      (distance(input.world_pos, uniforms.attenuation_color.xyz) - uniforms.physical_params3.z) / max(uniforms.physical_params3.w - uniforms.physical_params3.z, 0.0001),
      0.0,
      1.0,
    );
    return output_color(vec3<f32>(distance_depth, 0.0, 0.0), alpha);
  }

  let use_specular = shading_model == 0u;
  let use_phong = shading_model == 5u;
  let use_toon = shading_model == 7u;
  let use_shadow_material = shading_model == 9u;

  let mr_sample = textureSample(t_metallic_roughness, s_metallic_roughness, transform_metallic_roughness_map_uv(uv, uv2));
  let metallic = uniforms.metallic * mr_sample.b;
  let roughness = max(uniforms.roughness * mr_sample.g, 0.04);
  let clearcoat_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_clearcoat_map_uv(uv, uv2), 0).r;
  let clearcoat_roughness_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_clearcoat_roughness_map_uv(uv, uv2), 0).g;
  let transmission_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_transmission_map_uv(uv, uv2), 0).b;
  let thickness_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_thickness_map_uv(uv, uv2), 0).a;
  let sheen_color_sample = textureSample(t_physical_sheen, s_physical_sheen_map, transform_sheen_color_map_uv(uv, uv2)).rgb;
  let sheen_roughness_sample = textureSample(t_physical_sheen, s_physical_sheen_map, transform_sheen_roughness_map_uv(uv, uv2)).a;
  let physical_anisotropy_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_anisotropy_map_uv(uv, uv2), 1);
  let physical_specular_color_sample = textureSample(t_physical_specular, s_physical_specular_map, transform_specular_color_map_uv(uv, uv2)).rgb;
  let physical_specular_intensity_sample = textureSample(t_physical_specular, s_physical_specular_map, transform_specular_intensity_map_uv(uv, uv2)).a;
  let clearcoat = clamp(uniforms.physical_params1.x * clearcoat_sample, 0.0, 1.0);
  let clearcoat_roughness = max(uniforms.physical_params1.y * clearcoat_roughness_sample, 0.0525);
  let transmission = clamp(uniforms.physical_params1.z * transmission_sample, 0.0, 1.0);
  let ior = clamp(uniforms.physical_params1.w, 1.0, 2.333);
  let sheen_color = clamp(uniforms.physical_params2.rgb * sheen_color_sample, vec3<f32>(0.0), vec3<f32>(1.0));
  let sheen_roughness = clamp(uniforms.physical_params2.w * sheen_roughness_sample, 0.0001, 1.0);
  let anisotropy = clamp(uniforms.physical_params3.x, 0.0, 1.0);
  let anisotropy_rotation = uniforms.physical_params3.y;
  let thickness = max(uniforms.physical_params3.z * thickness_sample, 0.0);
  let attenuation_distance = max(uniforms.physical_params3.w, 0.0);

  if use_shadow_material {
    let shadow_alpha = alpha * (1.0 - sample_shadow(input.world_pos, N));
    let mapped_shadow = apply_output_color_space(aces_filmic_tone_mapping(albedo));
    let fogged_shadow = apply_fog(mapped_shadow, distance(input.world_pos, uniforms.camera_pos.xyz));
    return output_color(fogged_shadow, shadow_alpha);
  }
  let T = normalize(tbn[0]);
  let B = normalize(tbn[1]);
  let clearcoat_normal_sample = textureSample(t_clearcoat_normal, s_clearcoat_normal_map, transform_clearcoat_normal_map_uv(uv, uv2)).rgb;
  var clearcoat_tangent_normal = clearcoat_normal_sample * 2.0 - vec3<f32>(1.0);
  clearcoat_tangent_normal.x *= uniforms.physical_params4.x;
  clearcoat_tangent_normal.y *= uniforms.physical_params4.y;
  let Ncc = normalize(tbn * clearcoat_tangent_normal);
  let anisotropy_map_raw = physical_anisotropy_sample.rg * 2.0 - vec2<f32>(1.0);
  let anisotropy_map_dir = select(
    vec2<f32>(1.0, 0.0),
    normalize(anisotropy_map_raw),
    dot(anisotropy_map_raw, anisotropy_map_raw) > 0.0001,
  );
  let anisotropy_rot_c = cos(anisotropy_rotation);
  let anisotropy_rot_s = sin(anisotropy_rotation);
  let anisotropy_vec = vec2<f32>(
    anisotropy_rot_c * anisotropy_map_dir.x - anisotropy_rot_s * anisotropy_map_dir.y,
    anisotropy_rot_s * anisotropy_map_dir.x + anisotropy_rot_c * anisotropy_map_dir.y,
  ) * anisotropy * physical_anisotropy_sample.b;
  let anisotropy_strength = clamp(length(anisotropy_vec), 0.0, 1.0);
  let anisotropy_dir = select(
    vec2<f32>(1.0, 0.0),
    anisotropy_vec / max(anisotropy_strength, 0.0001),
    anisotropy_strength > 0.0001,
  );

  let V = normalize(uniforms.camera_pos.xyz - input.world_pos);
  let n_dot_v = max(dot(N, V), 0.0);

  // Dielectric F0 from IOR (1.5 -> 0.04), modulated by MeshPhysicalMaterial specular extensions.
  let dielectric_f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
  let physical_specular_color = clamp(uniforms.physical_specular.rgb * physical_specular_color_sample, vec3<f32>(0.0), vec3<f32>(1.0));
  let physical_specular_intensity = clamp(uniforms.physical_specular.w * physical_specular_intensity_sample, 0.0, 1.0);
  let dielectric_f0 = min(vec3<f32>(dielectric_f0_scalar) * physical_specular_color, vec3<f32>(1.0)) * physical_specular_intensity;
  let specular_f90 = mix(physical_specular_intensity, 1.0, metallic);
  let f0 = mix(dielectric_f0, albedo, metallic);
  let phong_specular_color = clamp(uniforms.physical_params2.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let phong_shininess = max(uniforms.physical_params2.w, 0.0001);
  var phong_specular_strength = 1.0;
  if use_phong && uniforms.physical_params4.w > 0.5 {
    phong_specular_strength = textureSample(t_physical_layers, s_specular_map, transform_specular_map_uv(uv2), 0).r;
  }

  var lo = vec3<f32>(0.0);

  let has_ibl = uniforms.normal_map_params.w > 0.5;
  let has_light_probe = uniforms.light_probe_params.x > 0.5;
  var light_probe_diffuse = vec3<f32>(0.0);
  if has_light_probe {
    light_probe_diffuse = light_probe_irradiance(N);
  }

  if uniforms.num_lights == 0u && !has_ibl && !has_light_probe {
    // No lights or IBL: render with a basic hemispherical ambient
    let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity;
    let sky_factor = 0.5 + 0.5 * N.y;
    let fallback_ambient = mix(vec3<f32>(0.1, 0.1, 0.12), vec3<f32>(0.4, 0.45, 0.5), sky_factor);
    let total_ambient = max(ambient, fallback_ambient);
    lo = albedo * total_ambient * ao + light_map_diffuse;
  } else {
    // Direct lighting from scene lights
    let shadow_factor = sample_shadow(input.world_pos, N);
    let shadow_light_index = u32(uniforms.shadow_params2.x);
    for (var i = 0u; i < uniforms.num_lights && i < MAX_LIGHTS; i = i + 1u) {
      let light = uniforms.lights[i];

      if light.light_type == 3u {
        // Hemisphere light
        let up = normalize(light.direction.xyz);
        let sky_color = light.color_intensity.rgb * light.color_intensity.w;
        let ground_color = light.position.xyz * light.color_intensity.w;
        let hemi_factor = 0.5 + 0.5 * dot(N, up);
        lo = lo + albedo * mix(ground_color, sky_color, hemi_factor);
        continue;
      }

      var L: vec3<f32>;
      var attenuation: f32 = 1.0;

      if light.light_type == 0u {
        // Directional
        L = normalize(-light.direction.xyz);
        if i == shadow_light_index {
          attenuation *= shadow_factor;
        }
      } else if light.light_type == 4u {
        // RectAreaLight approximation: finite one-sided area emitter from the
        // light center. This is intentionally cheaper than Three.js' LUT path.
        let light_vec = light.position.xyz - input.world_pos;
        let dist = length(light_vec);
        L = light_vec / max(dist, 0.0001);
        let width = max(light.params.x, 0.0);
        let height = max(light.params.y, 0.0);
        let area = max(width * height, 0.0001);
        let light_dir = normalize(light.direction.xyz);
        let facing = max(dot(light_dir, -L), 0.0);
        attenuation = facing * area / max(dist * dist + area, 0.0001);
      } else {
        // Point or Spot
        let light_vec = light.position.xyz - input.world_pos;
        let dist = length(light_vec);
        L = light_vec / max(dist, 0.0001);
        let cutoff_distance = light.position.w;
        let decay_exponent = light.direction.w;
        attenuation = get_distance_attenuation(dist, cutoff_distance, decay_exponent);

        // Spot cone attenuation
        if light.light_type == 2u {
          let cos_angle = dot(normalize(-light_vec), normalize(light.direction.xyz));
          let cone_cos = light.params.x;
          let penumbra_cos = light.params.y;
          attenuation *= get_spot_attenuation(cone_cos, penumbra_cos, cos_angle);
        }
        if i == shadow_light_index {
          attenuation *= shadow_factor;
        }
      }

      let H = normalize(V + L);
      let n_dot_l = max(dot(N, L), 0.0);
      let n_dot_h = max(dot(N, H), 0.0);
      let h_dot_v = max(dot(H, V), 0.0);

      let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

      if use_specular {
        // Cook-Torrance BRDF (MeshStandardMaterial / MeshPhysicalMaterial)
        var D = distribution_ggx(n_dot_h, roughness);
        var G = geometry_smith(n_dot_v, n_dot_l, roughness);
        if anisotropy_strength > 0.0001 {
          let anisotropy_t = normalize(T * anisotropy_dir.x + B * anisotropy_dir.y);
          let anisotropy_b = normalize(B * anisotropy_dir.x - T * anisotropy_dir.y);
          let alpha_b = max(roughness * roughness, 0.0001);
          let alpha_t = mix(alpha_b, 1.0, anisotropy_strength * anisotropy_strength);
          D = distribution_ggx_anisotropic(
            alpha_t,
            alpha_b,
            n_dot_h,
            dot(anisotropy_t, H),
            dot(anisotropy_b, H),
          );
          G = geometry_smith_correlated_anisotropic(
            alpha_t,
            alpha_b,
            dot(anisotropy_t, V),
            dot(anisotropy_b, V),
            dot(anisotropy_t, L),
            dot(anisotropy_b, L),
            n_dot_v,
            n_dot_l,
          );
        }
        let F = fresnel_schlick_f90(h_dot_v, f0, specular_f90);

        let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);

        let k_s = F;
        let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

        var physical_specular = specular;
        if clearcoat > 0.0001 {
          let cc_n_dot_v = max(dot(Ncc, V), 0.0);
          let cc_n_dot_l = max(dot(Ncc, L), 0.0);
          let cc_n_dot_h = max(dot(Ncc, H), 0.0);
          let Dcc = distribution_ggx(cc_n_dot_h, clearcoat_roughness);
          let Gcc = geometry_smith(cc_n_dot_v, cc_n_dot_l, clearcoat_roughness);
          let Fcc = fresnel_schlick(h_dot_v, vec3<f32>(0.04));
          physical_specular += clearcoat * (Dcc * Gcc * Fcc) / (4.0 * cc_n_dot_v * cc_n_dot_l + 0.0001);
        }
        if max(max(sheen_color.r, sheen_color.g), sheen_color.b) > 0.0001 {
          physical_specular += brdf_sheen(L, V, N, sheen_color, sheen_roughness);
        }

        lo = lo + (k_d * albedo / PI + physical_specular) * radiance * n_dot_l;
      } else if use_phong {
        // MeshPhongMaterial: non-physical Blinn-Phong direct lighting.
        let phong_f = fresnel_schlick(h_dot_v, phong_specular_color);
        let phong_d = (phong_shininess * 0.5 + 1.0) * pow(n_dot_h, phong_shininess) / PI;
        let phong_specular = phong_f * (0.25 * phong_d) * phong_specular_strength;
        lo = lo + (albedo / PI + phong_specular) * radiance * n_dot_l;
      } else if use_toon {
        // MeshToonMaterial: gradientMap samples the red ramp channel at dot(N, L) * 0.5 + 0.5.
        let toon_coord = dot(N, L) * 0.5 + 0.5;
        var toon_irradiance: f32;
        if uniforms.light_probe_params.y > 0.5 {
          toon_irradiance = textureSample(t_physical_sheen, s_physical, vec2<f32>(toon_coord, 0.0)).r;
        } else {
          let toon_width = fwidth(toon_coord) * 0.5;
          toon_irradiance = mix(0.7, 1.0, smoothstep(0.7 - toon_width, 0.7 + toon_width, toon_coord));
        }
        lo = lo + toon_irradiance * albedo / PI * radiance;
      } else {
        // MeshLambertMaterial: diffuse-only
        lo = lo + albedo / PI * radiance * n_dot_l;
      }
    }

    // Image-Based Lighting (split-sum approximation)
    if has_ibl {
      let env_intensity = uniforms.ibl_params.x;

      // Diffuse IBL: irradiance cubemap
      let irradiance = textureSample(t_irradiance, s_ibl, N).rgb;

      if use_specular {
        let F_ibl = fresnel_schlick_roughness_f90(n_dot_v, f0, specular_f90, roughness);
        let k_s_ibl = F_ibl;
        let k_d_ibl = (vec3<f32>(1.0) - k_s_ibl) * (1.0 - metallic);
        let diffuse_ibl = k_d_ibl * irradiance * albedo;

        // Specular IBL: prefiltered env map + BRDF LUT
        let R = reflect(-V, N);
        let max_lod = 4.0; // PREFILTER_MIP_LEVELS - 1
        let prefiltered_color = textureSampleLevel(t_prefilter, s_ibl, R, roughness * max_lod).rgb;
        let brdf_sample = textureSample(t_brdf_lut, s_ibl, vec2<f32>(n_dot_v, roughness)).rg;
        var specular_ibl = prefiltered_color * (F_ibl * brdf_sample.x + brdf_sample.y);

        if clearcoat > 0.0001 {
          let Rcc = reflect(-V, Ncc);
          let cc_n_dot_v = max(dot(Ncc, V), 0.0);
          let clearcoat_radiance = textureSampleLevel(t_prefilter, s_ibl, Rcc, clearcoat_roughness * max_lod).rgb;
          let brdf_clearcoat = textureSample(t_brdf_lut, s_ibl, vec2<f32>(cc_n_dot_v, clearcoat_roughness)).rg;
          let F_clearcoat = fresnel_schlick_roughness(cc_n_dot_v, vec3<f32>(0.04), clearcoat_roughness);
          specular_ibl += clearcoat * clearcoat_radiance * (F_clearcoat * brdf_clearcoat.x + brdf_clearcoat.y);
        }
        if max(max(sheen_color.r, sheen_color.g), sheen_color.b) > 0.0001 {
          specular_ibl += irradiance * sheen_color * ibl_sheen_brdf(N, V, sheen_roughness) / PI;
        }

        lo = lo + (diffuse_ibl + specular_ibl) * env_intensity * ao;
      } else {
        // Lambert: diffuse IBL only
        lo = lo + irradiance * albedo * env_intensity * ao;
      }
    } else {
      // Ambient (non-IBL fallback when lights are present)
      let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity * albedo;
      lo = lo + ambient * ao;
    }
    if has_light_probe {
      lo = lo + albedo * light_probe_diffuse * ao;
    }
    lo = lo + light_map_diffuse;
  }

  if use_specular && transmission > 0.0001 {
    let refracted_dir = refract(-V, N, 1.0 / ior);
    let transmittance = volume_attenuation(thickness, uniforms.attenuation_color.rgb, attenuation_distance);
    let scene_offset = refracted_dir.xy * thickness * 0.04;
    let scene_uv = clamp(screen_uv + scene_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    var transmitted_light = textureSample(t_scene_color, s_scene_color, scene_uv).rgb * transmittance;
    if has_ibl {
      let max_lod = 4.0;
      let environment_refraction = textureSampleLevel(t_prefilter, s_ibl, refracted_dir, roughness * max_lod).rgb * transmittance;
      transmitted_light = mix(transmitted_light, environment_refraction, 0.35);
    }
    lo = mix(lo, transmitted_light, transmission);
  }

  // Emissive
  let emissive_sample = decode_emissive_map_sample(textureSample(t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2))).rgb;
  lo = lo + uniforms.emissive.rgb * emissive_sample;

  // Tone mapping (ACES Filmic, matches three.js) and output color conversion.
  let mapped = aces_filmic_tone_mapping(lo);
  let output_mapped = apply_output_color_space(mapped);
  let fogged = apply_fog(output_mapped, distance(input.world_pos, uniforms.camera_pos.xyz));

  return output_color(fogged, alpha);
}

// ACES Filmic tone mapping, ported from three.js (Narkowicz fit with
// input/output matrices). Includes the 1/0.6 exposure pre-scale that
// three.js applies so output matches ACESFilmicToneMapping there.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
  let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
  let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
  return a / b;
}

fn aces_filmic_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
  // WGSL mat3x3 constructor takes columns.
  let aces_input = mat3x3<f32>(
    vec3<f32>(0.59719, 0.07600, 0.02840),
    vec3<f32>(0.35458, 0.90834, 0.13383),
    vec3<f32>(0.04823, 0.01566, 0.83777),
  );
  let aces_output = mat3x3<f32>(
    vec3<f32>( 1.60475, -0.10208, -0.00327),
    vec3<f32>(-0.53108,  1.10813, -0.07276),
    vec3<f32>(-0.07367, -0.00605,  1.07602),
  );
  var color = color_in * (1.0 / 0.6);
  color = aces_input * color;
  color = rrt_and_odt_fit(color);
  color = aces_output * color;
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_output_color_space(color: vec3<f32>) -> vec3<f32> {
  if uniforms.output_params.x > 0.5 {
    return color;
  }
  return pow(color, vec3<f32>(1.0 / 2.2));
}
"#;

pub fn custom_shader_source(fragment_body: &str) -> String {
    CUSTOM_FRAGMENT_SHADER.replace("__CUSTOM_FRAGMENT_BODY__", fragment_body)
}

const CUSTOM_FRAGMENT_SHADER: &str = r#"
const MAX_LIGHTS: u32 = 16u;
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
  ao_params: vec4<f32>,
  render_params: vec4<f32>,
  output_params: vec4<f32>,
  // texture_transform1.w = base texture uses secondary UV stream.
  // texture_transform2.w = base texture is sRGB and must be decoded to linear before shading.
  texture_transform1: vec4<f32>,
  texture_transform2: vec4<f32>,
  alpha_map_transform1: vec4<f32>,
  alpha_map_transform2: vec4<f32>,
  map_transform_rows: array<vec4<f32>, 12>,
  physical_map_transform_rows: array<vec4<f32>, 20>,
  clipping_planes: array<vec4<f32>, 8>,
  // x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
  clipping_params: vec4<f32>,
  fog_color: vec4<f32>,
  fog_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 6>,
  shadow_params: vec4<f32>,
  shadow_params2: vec4<f32>,
  shadow_params3: vec4<f32>,
  physical_params1: vec4<f32>,
  physical_params2: vec4<f32>,
  physical_params3: vec4<f32>,
  physical_params4: vec4<f32>,
  attenuation_color: vec4<f32>,
  physical_specular: vec4<f32>,
  lights: array<GpuLight, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

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

fn alpha_hash_threshold(position: vec4<f32>) -> f32 {
  let pixel = floor(position.xy);
  return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  if is_clipped_by_planes(input.world_pos) {
    discard;
  }

  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let uv2 = vec2<f32>(input.uv2.x, 1.0 - input.uv2.y);
  let map_uv = select(uv, uv2, uniforms.texture_transform1.w > 0.5);
  let uv1 = vec3<f32>(map_uv, 1.0);
  let transformed_uv = vec2<f32>(dot(uniforms.texture_transform1.xyz, uv1), dot(uniforms.texture_transform2.xyz, uv1));
  let texture_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, transformed_uv));
  let base_color = texture_color * input.color * uniforms.base_color;
  let alpha = base_color.a;
  let alpha_cutoff = uniforms.emissive.w;
  if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
    discard;
  }
  if uniforms.clipping_params.z > 0.5 && alpha < alpha_hash_threshold(input.position) {
    discard;
  }
  var normal = normalize(input.world_normal);
  if !front_facing {
    normal = -normal;
  }
  __CUSTOM_FRAGMENT_BODY__
}
"#;

pub const POST_SHADER: &str = r#"
struct PostUniforms {
  // x = exposure stops, y = contrast, z = saturation, w = vignette
  params1: vec4<f32>,
  // x = grayscale blend, y = invert blend
  params2: vec4<f32>,
};

@group(0) @binding(0)
var t_color: texture_2d<f32>;
@group(0) @binding(1)
var s_color: sampler;
@group(0) @binding(2)
var<uniform> uniforms: PostUniforms;

struct PostVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) vertex_index: u32) -> PostVertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  let pos = positions[vertex_index];
  var out: PostVertexOutput;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

@fragment
fn fs_post(input: PostVertexOutput) -> @location(0) vec4<f32> {
  let sample = textureSample(t_color, s_color, input.uv);
  var color = sample.rgb;

  color = color * exp2(uniforms.params1.x);
  color = (color - vec3<f32>(0.5)) * max(uniforms.params1.y, 0.0) + vec3<f32>(0.5);

  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = mix(vec3<f32>(luma), color, max(uniforms.params1.z, 0.0));
  color = mix(color, vec3<f32>(luma), clamp(uniforms.params2.x, 0.0, 1.0));

  if uniforms.params1.w > 0.0 {
    let centered = input.uv * 2.0 - vec2<f32>(1.0);
    let radius = dot(centered, centered);
    let vignette = mix(1.0, smoothstep(1.35, 0.15, radius), clamp(uniforms.params1.w, 0.0, 1.0));
    color *= vignette;
  }

  color = mix(color, vec3<f32>(1.0) - color, clamp(uniforms.params2.y, 0.0, 1.0));
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), sample.a);
}
"#;

pub const BACKGROUND_SHADER: &str = r#"
struct BackgroundUniforms {
  // transform1.xyz / transform2.xyz = texture transform rows.
  // transform1.w = background intensity.
  transform1: vec4<f32>,
  // transform2.w integer flags: +1 = texture is sRGB, +2 = LinearSRGBColorSpace output.
  // transform2.w fractional lane stores 2D background blur amount / 4.
  transform2: vec4<f32>,
};

@group(0) @binding(0)
var t_background: texture_2d<f32>;
@group(0) @binding(1)
var s_background: sampler;
@group(0) @binding(2)
var<uniform> uniforms: BackgroundUniforms;

struct BackgroundVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) vertex_index: u32) -> BackgroundVertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  let pos = positions[vertex_index];
  var out: BackgroundVertexOutput;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

fn transform_background_uv(uv: vec2<f32>) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(dot(uniforms.transform1.xyz, uv1), dot(uniforms.transform2.xyz, uv1));
}

fn background_srgb_to_linear_channel(value: f32) -> f32 {
  if value <= 0.04045 {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn background_srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    background_srgb_to_linear_channel(color.r),
    background_srgb_to_linear_channel(color.g),
    background_srgb_to_linear_channel(color.b),
  );
}

fn background_texture_is_srgb() -> bool {
  let flags = floor(uniforms.transform2.w);
  let srgb_flag = flags - floor(flags * 0.5) * 2.0;
  return srgb_flag > 0.5;
}

fn apply_background_output_color_space(color: vec3<f32>) -> vec3<f32> {
  if floor(uniforms.transform2.w) > 1.5 {
    return color;
  }
  return pow(color, vec3<f32>(1.0 / 2.2));
}

fn background_blur_amount() -> f32 {
  return fract(uniforms.transform2.w) * 4.0;
}

fn sample_background(uv: vec2<f32>) -> vec4<f32> {
  let transformed_uv = transform_background_uv(uv);
  let blur = background_blur_amount();
  if blur <= 0.001 {
    return textureSample(t_background, s_background, transformed_uv);
  }

  let dimensions = vec2<f32>(textureDimensions(t_background, 0));
  let texel = vec2<f32>(1.0) / max(dimensions, vec2<f32>(1.0));
  let offset = texel * (1.0 + blur * 8.0);
  var color = textureSample(t_background, s_background, transformed_uv) * 0.25;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x, 0.0)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x, 0.0)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(0.0,  offset.y)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(0.0, -offset.y)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x,  offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x,  offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x, -offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x, -offset.y)) * 0.0625;
  return color;
}

@fragment
fn fs_background(input: BackgroundVertexOutput) -> @location(0) vec4<f32> {
  let sample = sample_background(input.uv);
  var color = sample.rgb;
  if background_texture_is_srgb() {
    color = background_srgb_to_linear(color);
  }
  color *= uniforms.transform1.w;
  color = apply_background_output_color_space(color);
  return vec4<f32>(color, sample.a);
}
"#;
