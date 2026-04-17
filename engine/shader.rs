pub const SHADER: &str = r#"
const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 16u;

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
  // spot: x = cos(outer_angle), y = cos(inner_angle)
  params: vec4<f32>,
};

struct Uniforms {
  mvp: mat4x4<f32>,
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
  // x = normal_scale.x, y = normal_scale.y, z = has_normal_map (1.0 or 0.0)
  normal_map_params: vec4<f32>,
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

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec4<f32>,
  @location(3) color: vec4<f32>,
  @location(4) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) world_tangent: vec3<f32>,
  @location(3) tangent_w: f32,
  @location(4) color: vec4<f32>,
  @location(5) uv: vec2<f32>,
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
  return output;
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

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Three.js-compatible distance attenuation
// distance = cutoff distance (0 = infinite range)
// decay = decay exponent (default 2, physically correct)
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let tex_color = textureSample(t_diffuse, s_diffuse, uv);
  let albedo = tex_color.rgb * input.color.rgb * uniforms.base_color.rgb;
  let alpha = tex_color.a * input.color.a * uniforms.base_color.a;

  let mr_sample = textureSample(t_metallic_roughness, s_metallic_roughness, uv);
  let metallic = uniforms.metallic * mr_sample.b;
  let roughness = max(uniforms.roughness * mr_sample.g, 0.04);

  // Normal mapping via TBN matrix
  var N = normalize(input.world_normal);
  if uniforms.normal_map_params.z > 0.5 {
    let T = normalize(input.world_tangent);
    let B = normalize(cross(N, T) * input.tangent_w);
    let tbn = mat3x3<f32>(T, B, N);
    let normal_sample = textureSample(t_normal, s_normal, uv).rgb;
    var tangent_normal = normal_sample * 2.0 - vec3<f32>(1.0);
    tangent_normal.x *= uniforms.normal_map_params.x;
    tangent_normal.y *= uniforms.normal_map_params.y;
    N = normalize(tbn * tangent_normal);
  }

  let V = normalize(uniforms.camera_pos.xyz - input.world_pos);
  let n_dot_v = max(dot(N, V), 0.0);

  // Dielectric F0 = 0.04, metallic F0 = albedo
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);

  var lo = vec3<f32>(0.0);

  if uniforms.num_lights == 0u {
    // No lights: render with a basic hemispherical ambient
    let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity;
    let sky_factor = 0.5 + 0.5 * N.y;
    let fallback_ambient = mix(vec3<f32>(0.1, 0.1, 0.12), vec3<f32>(0.4, 0.45, 0.5), sky_factor);
    let total_ambient = max(ambient, fallback_ambient);
    lo = albedo * total_ambient;
  } else {
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
      }

      let H = normalize(V + L);
      let n_dot_l = max(dot(N, L), 0.0);
      let n_dot_h = max(dot(N, H), 0.0);
      let h_dot_v = max(dot(H, V), 0.0);

      // Cook-Torrance BRDF
      let D = distribution_ggx(n_dot_h, roughness);
      let G = geometry_smith(n_dot_v, n_dot_l, roughness);
      let F = fresnel_schlick(h_dot_v, f0);

      let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);

      let k_s = F;
      let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

      let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;
      lo = lo + (k_d * albedo / PI + specular) * radiance * n_dot_l;
    }

    // Ambient
    let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity * albedo;
    lo = lo + ambient;
  }

  // Emissive
  lo = lo + uniforms.emissive.rgb;

  // Tone mapping (Reinhard) and gamma correction
  let mapped = lo / (lo + vec3<f32>(1.0));
  let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(gamma_corrected, alpha);
}
"#;
