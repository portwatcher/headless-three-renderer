// MToon lighting follows Pixiv three-vrm 3.4.4's WebGL shader (MIT).
// https://github.com/pixiv/three-vrm/tree/v3.4.4/packages/three-vrm-materials-mtoon
// Authored shade colors are a separate surface, not a multiplier on lit albedo.
fn mtoon_lighting(input: VertexOutput, N: vec3<f32>, albedo: vec3<f32>, uv: vec2<f32>, uv2: vec2<f32>) -> vec3<f32> {
  if uniforms.mtoon[5].x > 0.5 && uniforms.mtoon[4].w == 0.0 {
    return uniforms.mtoon[4].rgb;
  }
  let shade = uniforms.mtoon[0].rgb * input.color.rgb * decode_matcap_map_sample(textureSample(
    t_physical_sheen, s_physical_sheen_map, transform_matcap_color_map_uv(uv, uv2)
  )).rgb;
  let shift_sample = decode_metallic_roughness_map_sample(textureSample(
    t_metallic_roughness, s_metallic_roughness, transform_metallic_roughness_map_uv(uv, uv2)
  )).r;
  let shift = uniforms.mtoon[0].w + shift_sample * uniforms.mtoon[3].y;
  let edge = max(1.0 - uniforms.mtoon[3].x, 0.00001);
  var indirect = uniforms.ambient_color.rgb * uniforms.ambient_intensity;
  if uniforms.light_probe_params.x > 0.5 {
    indirect += light_probe_irradiance(N);
  }
  var direct = vec3<f32>(0.0);
  var rim_irradiance = vec3<f32>(0.0);
  for (var i = 0u; i < uniforms.num_lights && i < MAX_LIGHTS; i += 1u) {
    let light = uniforms.lights[i];
    let radiance = light.color_intensity.rgb * light.color_intensity.w;
    if light.light_type == 3u {
      let ground = light.position.xyz * light.color_intensity.w;
      indirect += mix(ground, radiance, 0.5 + 0.5 * dot(N, normalize(light.direction.xyz)));
      continue;
    }
    // Pixiv MToon does not implement RectAreaLight.
    if light.light_type == 4u {
      continue;
    }
    var L = normalize(-light.direction.xyz);
    var attenuation = 1.0;
    if light.light_type != 0u {
      let delta = light.position.xyz - input.world_pos;
      let distance = length(delta);
      L = delta / max(distance, 0.0001);
      attenuation = get_distance_attenuation(distance, light.position.w, light.direction.w);
      if light.light_type == 2u {
        attenuation *= get_spot_attenuation(light.params.x, light.params.y, dot(-L, normalize(light.direction.xyz)));
      }
    }
    let incident = radiance * attenuation;
    let shadow = sample_shadow_for_light(i, input.world_pos, N);
    let shading = clamp((dot(N, L) + shift + edge) / (2.0 * edge), 0.0, 1.0) * shadow;
    var diffuse = mix(shade, albedo, shading) * incident / PI;
    if uniforms.mtoon[3].z > 0.5 {
      diffuse = min(diffuse, albedo);
    }
    direct += diffuse;
    rim_irradiance += incident;
  }
  // Unlike the generic native toon path, ambient irradiance is also Lambert-normalized.
  var color = direct + indirect * albedo / PI;
  rim_irradiance += indirect;
  let view_position = (uniforms.view * vec4<f32>(input.world_pos, 1.0)).xyz;
  let V = normalize(-view_position);
  let view_normal = normalize((uniforms.view * vec4<f32>(N, 0.0)).xyz);
  let rim = uniforms.mtoon[1].rgb * pow(
    clamp(1.0 - dot(V, view_normal) + uniforms.mtoon[2].w, 0.0, 1.0), uniforms.mtoon[1].w
  );
  let x = normalize(vec3<f32>(V.z, 0.0, -V.x));
  let y = cross(V, x);
  let matcap_uv = 0.5 + 0.5 * vec2<f32>(dot(x, view_normal), -dot(y, view_normal));
  let matcap = uniforms.mtoon[2].rgb * decode_light_map_sample(textureSample(
    t_light_map, s_light_map, transform_light_map_uv(matcap_uv, matcap_uv)
  )).rgb;
  var rim_texture = textureSample(t_physical_layers, s_specular_map, transform_specular_map_uv(uv, uv2), 0).rgb;
  if uniforms.map_transform_rows[10u].w > 0.5 {
    rim_texture = srgb_to_linear(rim_texture);
  }
  // Pixiv's current WebGL shader uses lighting mix 1 and divides irradiance by PI.
  color += (rim + matcap) * rim_texture * rim_irradiance / PI;
  color += uniforms.emissive.rgb * decode_emissive_map_sample(textureSample(
    t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2)
  )).rgb;
  if uniforms.mtoon[5].x > 0.5 {
    color = uniforms.mtoon[4].rgb * mix(vec3<f32>(1.0), color, uniforms.mtoon[4].w);
  }
  return color;
}
