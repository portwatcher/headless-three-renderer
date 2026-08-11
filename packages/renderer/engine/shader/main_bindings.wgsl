const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 64u;
const MAX_SHADOW_LAYERS: u32 = 12u;
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
  // x/y = normalScale or bumpScale, z = normal mode (0=none, 1=tangent normalMap, 2=bumpMap, 3=object normalMap), w = has_ibl
  normal_map_params: vec4<f32>,
  // x = env_intensity, y = shading_model (0=standard PBR, 1=basic/unlit, 2=lambert, 3=normal, 4=matcap, 5=phong, 6=depth, 7=toon, 8=distance, 9=shadow), z = camera near, w = camera far
  ibl_params: vec4<f32>,
  // x = legacy env combine (0=multiply, 1=mix, 2=add), y = reflectivity, z = material env mode (0=off, 1=reflect, 2=refract), w = refraction ratio
  env_map_params: vec4<f32>,
  // x = ao_map_intensity, y = has_ao_map, z = has_alpha_map, w = has_light_map
  ao_params: vec4<f32>,
  // x = 1/width, y = 1/height, z = width, w = height
  render_params: vec4<f32>,
  // x = 1 for LinearSRGBColorSpace output, 0 for SRGBColorSpace output; y = tone-mapping mode (0 when material toneMapped=false); z = alpha-to-coverage active; w = toneMappingExposure.
  output_params: vec4<f32>,
  // texture_transform1.xyz / texture_transform2.xyz = base-color texture transform rows.
  // texture_transform1.w = base texture uses secondary UV stream.
  // texture_transform2.w = base texture is sRGB and must be decoded to linear before shading.
  texture_transform1: vec4<f32>,
  texture_transform2: vec4<f32>,
  // alpha_map_transform1.xyz / alpha_map_transform2.xyz = alpha-map texture transform rows.
  // alpha_map_transform1.w = alpha map is sRGB.
  // alpha_map_transform2.w = alpha map uses secondary UV stream.
  alpha_map_transform1: vec4<f32>,
  alpha_map_transform2: vec4<f32>,
  // Row pairs for normal, metallic-roughness, emissive, AO, light, and specular map transforms.
  // map_transform_rows[1].w = active normal/bump map uses secondary UV stream.
  // map_transform_rows[2].w = metallic-roughness map is sRGB.
  // map_transform_rows[3].w = metallic-roughness map uses secondary UV stream.
  // map_transform_rows[4].w = emissive map is sRGB.
  // map_transform_rows[5].w = emissive map uses secondary UV stream.
  // map_transform_rows[6].w = AO map is sRGB.
  // map_transform_rows[7].w = AO map uses secondary UV stream.
  // map_transform_rows[8].w = light map is sRGB.
  // map_transform_rows[9].w = light map uses secondary UV stream.
  // map_transform_rows[10].w = Phong specular map is sRGB.
  // map_transform_rows[11].w = Phong specular map uses secondary UV stream.
  map_transform_rows: array<vec4<f32>, 12>,
  // Row pairs for current physical-extension map transforms.
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
  // physical_map_transform_rows[21].w = iridescence map uses secondary UV stream.
  // physical_map_transform_rows[23].w = iridescence thickness map uses secondary UV stream.
  physical_map_transform_rows: array<vec4<f32>, 24>,
  // World-space clipping planes [normal.xyz, constant].
  clipping_planes: array<vec4<f32>, 8>,
  // x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
  clipping_params: vec4<f32>,
  // xyz = fog color
  fog_color: vec4<f32>,
  // x = mode (0=off, 1=linear, 2=exp2), y = near, z = far, w = density
  fog_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 12>,
  // x = shadow count, y = first bias, z = first normal_bias, w = receive_shadow
  shadow_params: vec4<f32>,
  // x = first shadow light index, y = 1/map_width, z = 1/map_height, w = first shadow kind (0=2D, 1=point, 2=cascaded)
  shadow_params2: vec4<f32>,
  // x/y/z = first cascade split distances, w = first shadow layer count.
  shadow_params3: vec4<f32>,
  // x = first PCF radius multiplier, y = clip shadow caster fragments by clipping_planes, z = explicit shadow side (0=double/no-cull, 1=front, 2=back), w = shadow-only alpha cutoff.
  shadow_params4: vec4<f32>,
  // x = light index, y = layer base, z = layer count, w = shadow kind.
  shadow_infos: array<vec4<f32>, 12>,
  // x = bias, y = normal_bias, z = PCF radius multiplier, w = shadow intensity.
  shadow_biases: array<vec4<f32>, 12>,
  // x/y/z = cascade split distances, w = reserved.
  shadow_cascade_splits: array<vec4<f32>, 12>,
  // x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
  physical_params1: vec4<f32>,
  // xyz = sheen color, w = sheen roughness
  physical_params2: vec4<f32>,
  // x = anisotropy, y = anisotropy rotation, z/w = thickness/attenuation distance or distance near/far
  physical_params3: vec4<f32>,
  // x/y = clearcoat normal scale, z = light_map_intensity, w = has_specular_map, matcap map sRGB flag, or toon gradient map sRGB flag depending on shading model.
  physical_params4: vec4<f32>,
  // xyz = attenuation color or distance reference position, w = dispersion for physical materials
  attenuation_color: vec4<f32>,
  // xyz = MeshPhysicalMaterial specular color factor, w = specular intensity.
  physical_specular: vec4<f32>,
  // x = iridescence, y = iridescence IOR, z/w = iridescence thickness range in nanometers.
  iridescence_params: vec4<f32>,
  lights: array<GpuLight, 64>,
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

struct ShadowVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) uv2: vec2<f32>,
  @location(3) color: vec4<f32>,
};

fn shadow_vertex(input: VertexInput, layer: u32) -> ShadowVertexOutput {
  var output: ShadowVertexOutput;
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.light_space_matrices[layer] * world_pos;
  output.world_pos = world_pos.xyz;
  output.uv = input.uv;
  output.uv2 = input.uv2;
  output.color = input.color;
  return output;
}

@vertex
fn vs_shadow0(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 0u);
}

@vertex
fn vs_shadow1(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 1u);
}

@vertex
fn vs_shadow2(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 2u);
}

@vertex
fn vs_shadow3(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 3u);
}

@vertex
fn vs_shadow4(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 4u);
}

@vertex
fn vs_shadow5(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 5u);
}

@vertex
fn vs_shadow6(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 6u);
}

@vertex
fn vs_shadow7(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 7u);
}

@vertex
fn vs_shadow8(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 8u);
}

@vertex
fn vs_shadow9(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 9u);
}

@vertex
fn vs_shadow10(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 10u);
}

@vertex
fn vs_shadow11(input: VertexInput) -> ShadowVertexOutput {
  return shadow_vertex(input, 11u);
}

@fragment
fn fs_shadow(input: ShadowVertexOutput, @builtin(front_facing) front_facing: bool) {
  let shadow_side = uniforms.shadow_params4.z;
  if shadow_side > 0.5 {
    if shadow_side < 1.5 && !front_facing {
      discard;
    }
    if shadow_side > 1.5 && front_facing {
      discard;
    }
  }

  if uniforms.shadow_params4.y > 0.5 && is_clipped_by_planes(input.world_pos) {
    discard;
  }

  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let uv2 = vec2<f32>(input.uv2.x, 1.0 - input.uv2.y);
  var alpha = textureSample(t_diffuse, s_diffuse, transform_map_uv(uv, uv2)).a * input.color.a * uniforms.base_color.a;
  if uniforms.ao_params.z > 0.5 {
    let alpha_uv = select(uv, uv2, uniforms.alpha_map_transform2.w > 0.5);
    alpha = alpha * decode_alpha_map_sample(textureSample(t_alpha, s_alpha, transform_alpha_map_uv(alpha_uv)).g);
  }

  let alpha_cutoff = max(uniforms.emissive.w, uniforms.shadow_params4.w);
  if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
    discard;
  }
  if uniforms.clipping_params.z > 0.5 && alpha < alpha_hash_threshold(input.position) {
    discard;
  }
}

fn sample_shadow_layer(shadow_slot: u32, world_pos: vec3<f32>, layer: u32, world_normal: vec3<f32>) -> f32 {
  let bias = uniforms.shadow_biases[shadow_slot];
  let biased_pos = world_pos + world_normal * bias.y;
  let light_ndc = uniforms.light_space_matrices[layer] * vec4<f32>(biased_pos, 1.0);
  let proj = light_ndc.xyz / light_ndc.w;

  if proj.z > 1.0 || proj.z < 0.0 {
    return 1.0;
  }
  let uv = vec2<f32>(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
  if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
    return 1.0;
  }

  let reference = proj.z - bias.x;
  let texel = uniforms.shadow_params2.yz;
  let shadow_map_type = uniforms.shadow_params2.w;
  let intensity = max(bias.w, 0.0);

  if shadow_map_type < 0.5 {
    let visibility = textureSampleCompareLevel(t_shadow, s_shadow, uv, layer, reference);
    return mix(1.0, visibility, intensity);
  }

  // 3x3 PCF.
  var sum: f32 = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let offset = vec2<f32>(f32(dx), f32(dy)) * texel * bias.z;
      sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, uv + offset, layer, reference);
    }
  }
  return mix(1.0, sum / 9.0, intensity);
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

fn sample_shadow_slot(shadow_slot: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  let info = uniforms.shadow_infos[shadow_slot];
  let layer_base = u32(info.y);
  let layer_count = max(u32(info.z), 1u);
  let kind = info.w;
  if kind > 0.5 {
    if kind > 1.5 {
      let splits = uniforms.shadow_cascade_splits[shadow_slot];
      let camera_dist = distance(world_pos, uniforms.camera_pos.xyz);
      var local_layer = 0u;
      if camera_dist > splits.x {
        local_layer = 1u;
      }
      if camera_dist > splits.y {
        local_layer = 2u;
      }
      if camera_dist > splits.z {
        local_layer = 3u;
      }
      local_layer = min(local_layer, layer_count - 1u);
      return sample_shadow_layer(shadow_slot, world_pos, layer_base + local_layer, world_normal);
    }
    let light_index = u32(info.x);
    let light_pos = uniforms.lights[light_index].position.xyz;
    let local_layer = point_shadow_layer(world_pos - light_pos);
    return sample_shadow_layer(shadow_slot, world_pos, layer_base + local_layer, world_normal);
  }
  return sample_shadow_layer(shadow_slot, world_pos, layer_base, world_normal);
}

// 3x3 PCF shadow sampling. Returns the fraction of samples NOT in shadow
// for the requested light (i.e. 1.0 = fully lit, 0.0 = fully occluded).
fn sample_shadow_for_light(light_index: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  if uniforms.shadow_params.x < 0.5 || uniforms.shadow_params.w < 0.5 {
    return 1.0;
  }
  let shadow_count = min(u32(uniforms.shadow_params.x), MAX_SHADOW_LAYERS);
  for (var shadow_slot = 0u; shadow_slot < MAX_SHADOW_LAYERS; shadow_slot = shadow_slot + 1u) {
    if shadow_slot >= shadow_count {
      break;
    }
    if u32(uniforms.shadow_infos[shadow_slot].x) == light_index {
      return sample_shadow_slot(shadow_slot, world_pos, world_normal);
    }
  }
  return 1.0;
}

fn sample_combined_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  if uniforms.shadow_params.x < 0.5 || uniforms.shadow_params.w < 0.5 {
    return 1.0;
  }
  let shadow_count = min(u32(uniforms.shadow_params.x), MAX_SHADOW_LAYERS);
  var visibility = 1.0;
  for (var shadow_slot = 0u; shadow_slot < MAX_SHADOW_LAYERS; shadow_slot = shadow_slot + 1u) {
    if shadow_slot >= shadow_count {
      break;
    }
    visibility = min(visibility, sample_shadow_slot(shadow_slot, world_pos, world_normal));
  }
  return visibility;
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

fn apply_ior_to_roughness(roughness: f32, ior: f32) -> f32 {
  return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

fn iridescence_fresnel_color(n_dot_v: f32, ior: f32, thickness_min: f32, thickness_max: f32) -> vec3<f32> {
  let film_thickness = clamp((thickness_min + thickness_max) * 0.5, 0.0, 1200.0);
  let view_phase = pow(1.0 - clamp(n_dot_v, 0.0, 1.0), 1.5) * 1.35;
  let ior_phase = clamp(ior - 1.0, 0.0, 1.333) * 0.6;
  let phase = film_thickness * 0.006 + view_phase + ior_phase;
  let shifted = phase + vec3<f32>(0.00, 0.33, 0.67);
  return 0.5 + 0.5 * cos(shifted * 6.2831853);
}

fn sample_transmission_scene_color(scene_uv: vec2<f32>, roughness: f32, ior: f32) -> vec3<f32> {
  let clamped_uv = clamp(scene_uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let base = textureSample(t_scene_color, s_scene_color, clamped_uv).rgb;
  let transmission_roughness = apply_ior_to_roughness(roughness, ior);
  if transmission_roughness < 0.05 {
    return base;
  }
