  let dimensions = vec2<f32>(textureDimensions(t_scene_color, 0));
  let texel_size = 1.0 / max(dimensions, vec2<f32>(1.0));
  let lod_radius = exp2(log2(max(max(dimensions.x, dimensions.y), 2.0)) * transmission_roughness);
  let radius = max(1.0, lod_radius * 0.35);
  let offset = texel_size * radius;

  var color = base * 0.28;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>( offset.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.12;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>(-offset.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.12;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>(0.0,  offset.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.12;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>(0.0, -offset.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.12;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.06;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv - offset, vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.06;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>( offset.x, -offset.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.06;
  color += textureSample(t_scene_color, s_scene_color, clamp(clamped_uv + vec2<f32>(-offset.x,  offset.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb * 0.06;
  return color;
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

fn fog_depth(world_pos: vec3<f32>) -> f32 {
  let view_pos = uniforms.view * vec4<f32>(world_pos, 1.0);
  return max(-view_pos.z, 0.0);
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

fn transform_ao_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let ao_uv = select(uv, uv2, uniforms.map_transform_rows[7u].w > 0.5);
  return transform_slot_uv(ao_uv, 6u);
}

fn transform_light_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let light_uv = select(uv, uv2, uniforms.map_transform_rows[9u].w > 0.5);
  return transform_slot_uv(light_uv, 8u);
}

fn transform_specular_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let specular_uv = select(uv, uv2, uniforms.map_transform_rows[11u].w > 0.5);
  return transform_slot_uv(specular_uv, 10u);
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

fn transform_iridescence_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let iridescence_uv = select(uv, uv2, uniforms.physical_map_transform_rows[21u].w > 0.5);
  return transform_physical_slot_uv(iridescence_uv, 20u);
}

fn transform_iridescence_thickness_map_uv(uv: vec2<f32>, uv2: vec2<f32>) -> vec2<f32> {
  let thickness_uv = select(uv, uv2, uniforms.physical_map_transform_rows[23u].w > 0.5);
  return transform_physical_slot_uv(thickness_uv, 22u);
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

fn decode_metallic_roughness_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.map_transform_rows[2u].w > 0.5 {
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

fn decode_ao_map_sample(sample: f32) -> f32 {
  if uniforms.map_transform_rows[6u].w > 0.5 {
    return srgb_to_linear_channel(sample);
  }
  return sample;
}

fn decode_light_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.map_transform_rows[8u].w > 0.5 {
    return vec4<f32>(srgb_to_linear(sample.rgb), sample.a);
  }
  return sample;
}

fn decode_specular_map_sample(sample: f32) -> f32 {
  if uniforms.map_transform_rows[10u].w > 0.5 {
    return srgb_to_linear_channel(sample);
  }
  return sample;
}

fn decode_toon_gradient_map_sample(sample: vec4<f32>) -> vec4<f32> {
  if uniforms.physical_params4.w > 0.5 {
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
  let screen_uv = clamp(input.position.xy * uniforms.render_params.xy, vec2<f32>(0.0), vec2<f32>(1.0));
  let tex_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, transform_map_uv(uv, uv2)));
  let albedo = tex_color.rgb * input.color.rgb * uniforms.base_color.rgb;
  var alpha = tex_color.a * input.color.a * uniforms.base_color.a;
  if uniforms.ao_params.z > 0.5 {
    let alpha_uv = select(uv, uv2, uniforms.alpha_map_transform2.w > 0.5);
    alpha = alpha * decode_alpha_map_sample(textureSample(t_alpha, s_alpha, transform_alpha_map_uv(alpha_uv)).g);
  }
  alpha = alpha * clip_opacity;

  // Alpha test: discard fragments below the cutoff threshold
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
  } else if normal_mode == 3u {
    let normal_sample = textureSample(t_normal, s_normal, transform_normal_map_uv(uv, uv2)).rgb;
    var object_normal = normal_sample * 2.0 - vec3<f32>(1.0);
    if !front_facing {
      object_normal = -object_normal;
    }
    N = normalize((uniforms.normal_matrix * vec4<f32>(object_normal, 0.0)).xyz);
    tbn = tangent_basis(N, tbn[0], input.tangent_w);
  }

  // Ambient occlusion: sample red channel, blend toward 1.0 by intensity.
  // Matches three.js: ao = (texture.r - 1.0) * aoMapIntensity + 1.0
  var ao: f32 = 1.0;
  if uniforms.ao_params.y > 0.5 {
    let ao_sample = decode_ao_map_sample(textureSample(t_ao, s_ao, transform_ao_map_uv(uv, uv2)).r);
    ao = (ao_sample - 1.0) * uniforms.ao_params.x + 1.0;
  }
  let has_light_map = uniforms.ao_params.w > 0.5;
  var light_map_irradiance = vec3<f32>(0.0);
  if has_light_map {
    light_map_irradiance = decode_light_map_sample(textureSample(t_light_map, s_light_map, transform_light_map_uv(uv, uv2))).rgb * max(uniforms.physical_params4.z, 0.0);
  }
  let light_map_diffuse = albedo * light_map_irradiance * (1.0 / PI) * ao;

  // MeshBasicMaterial: unlit. Output = albedo * ao, then emissive + tone map + gamma.
  if shading_model == 1u {
    var unlit = albedo * ao;
    if has_light_map {
      unlit = light_map_diffuse;
    }
    let basic_env_mode = u32(uniforms.env_map_params.z + 0.5);
    if uniforms.normal_map_params.w > 0.5 && basic_env_mode > 0u {
      let V_basic = normalize(uniforms.camera_pos.xyz - input.world_pos);
      var env_dir_basic = reflect(-V_basic, N);
      if basic_env_mode == 2u {
        env_dir_basic = refract(-V_basic, N, uniforms.env_map_params.w);
      }
      let env_color = textureSampleLevel(t_prefilter, s_ibl, env_dir_basic, 0.0).rgb * uniforms.ibl_params.x;
      let reflectivity = uniforms.env_map_params.y;
      let combine = u32(uniforms.env_map_params.x + 0.5);
      if combine == 2u {
        unlit = unlit + env_color * reflectivity;
      } else if combine == 1u {
        unlit = mix(unlit, env_color, reflectivity);
      } else {
        unlit = mix(unlit, unlit * env_color, reflectivity);
      }
    }
    let emissive_basic = decode_emissive_map_sample(textureSample(t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2))).rgb;
    unlit = unlit + uniforms.emissive.rgb * emissive_basic;
    let mapped_basic = apply_output_color_space(apply_material_tone_mapping(unlit));
    let fogged_basic = apply_fog(mapped_basic, fog_depth(input.world_pos));
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
      let matcap_map = decode_matcap_map_sample(textureSample(t_physical_sheen, s_physical_sheen_map, transform_matcap_color_map_uv(uv, uv2)));
      matcap_surface_color *= matcap_map.rgb;
    }
    var matcap_color = decode_color_map_sample(textureSample(t_diffuse, s_diffuse, matcap_uv)).rgb * matcap_surface_color;
    let mapped_matcap = apply_output_color_space(apply_material_tone_mapping(matcap_color));
    let fogged_matcap = apply_fog(mapped_matcap, fog_depth(input.world_pos));
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
  let use_lambert = !use_specular && !use_phong && !use_toon && !use_shadow_material;
  let legacy_material_env = uniforms.env_map_params.z > 0.5 && (use_phong || use_lambert);
  let legacy_env_reflectivity = select(1.0, uniforms.env_map_params.y, legacy_material_env);

  let mr_sample = decode_metallic_roughness_map_sample(textureSample(t_metallic_roughness, s_metallic_roughness, transform_metallic_roughness_map_uv(uv, uv2)));
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
  let iridescence_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_iridescence_map_uv(uv, uv2), 2).r;
  let iridescence_thickness_sample = textureSample(t_physical_layers, s_physical_layers_map, transform_iridescence_thickness_map_uv(uv, uv2), 2).g;
  let clearcoat = clamp(uniforms.physical_params1.x * clearcoat_sample, 0.0, 1.0);
  let clearcoat_roughness = max(uniforms.physical_params1.y * clearcoat_roughness_sample, 0.0525);
  let transmission = clamp(uniforms.physical_params1.z * transmission_sample, 0.0, 1.0);
  let ior = clamp(uniforms.physical_params1.w, 1.0, 2.333);
  let sheen_color = clamp(uniforms.physical_params2.rgb * sheen_color_sample, vec3<f32>(0.0), vec3<f32>(1.0));
  let sheen_roughness = clamp(uniforms.physical_params2.w * sheen_roughness_sample, 0.0001, 1.0);
