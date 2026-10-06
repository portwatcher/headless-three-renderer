  let anisotropy = clamp(uniforms.physical_params3.x, 0.0, 1.0);
  let anisotropy_rotation = uniforms.physical_params3.y;
  let thickness = max(uniforms.physical_params3.z * thickness_sample, 0.0);
  let attenuation_distance = max(uniforms.physical_params3.w, 0.0);

  if use_shadow_material {
    let shadow_alpha = alpha * (1.0 - sample_combined_shadow(input.world_pos, N));
    let mapped_shadow = apply_output_color_space(apply_material_tone_mapping(albedo));
    let fogged_shadow = apply_fog(mapped_shadow, fog_depth(input.world_pos));
    return output_color(fogged_shadow, shadow_alpha);
  }
  // Three.js uses the tangent frame columns as they are (getTangentFrame scales them).
  let T = tbn[0];
  let B = tbn[1];
  let clearcoat_normal_sample = textureSample(t_clearcoat_normal, s_clearcoat_normal_map, transform_clearcoat_normal_map_uv(uv, uv2)).rgb;
  var clearcoat_tangent_normal = clearcoat_normal_sample * 2.0 - vec3<f32>(1.0);
  clearcoat_tangent_normal.x *= uniforms.physical_params4.x;
  clearcoat_tangent_normal.y *= uniforms.physical_params4.y;
  // Three.js clearcoatNormal: the non-perturbed normal, or the clearcoat normal map on its frame.
  let Ncc = select(geometry_normal, normalize(geometry_tbn * clearcoat_tangent_normal), uniforms.surface_params.z > 0.5);
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

  let V = view_direction(input.world_pos);
  let n_dot_v = max(dot(N, V), 0.0);

  // Dielectric F0 from IOR (1.5 -> 0.04), modulated by MeshPhysicalMaterial specular extensions.
  let dielectric_f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
  let physical_specular_color = clamp(uniforms.physical_specular.rgb * physical_specular_color_sample, vec3<f32>(0.0), vec3<f32>(1.0));
  let physical_specular_intensity = clamp(uniforms.physical_specular.w * physical_specular_intensity_sample, 0.0, 1.0);
  let dielectric_f0 = min(vec3<f32>(dielectric_f0_scalar) * physical_specular_color, vec3<f32>(1.0)) * physical_specular_intensity;
  let specular_f90 = mix(physical_specular_intensity, 1.0, metallic);
  let iridescence_strength = clamp(uniforms.iridescence_params.x * iridescence_sample, 0.0, 1.0) * (1.0 - metallic);
  let iridescence_thickness = mix(uniforms.iridescence_params.z, uniforms.iridescence_params.w, iridescence_thickness_sample);
  let iridescence_f0 = iridescence_fresnel_color(
    n_dot_v,
    clamp(uniforms.iridescence_params.y, 1.0, 2.333),
    iridescence_thickness,
    iridescence_thickness,
  ) * physical_specular_intensity;
  let f0 = mix(mix(dielectric_f0, iridescence_f0, iridescence_strength), albedo, metallic);
  let phong_specular_color = max(uniforms.physical_params2.rgb, vec3<f32>(0.0));
  let phong_shininess = max(uniforms.physical_params2.w, 0.0001);
  var phong_specular_strength = 1.0;
  if use_phong && uniforms.physical_params4.w > 0.5 {
    phong_specular_strength = decode_specular_map_sample(textureSample(t_physical_layers, s_specular_map, transform_specular_map_uv(uv, uv2), 0).r);
  }
  // Three.js material.anisotropyT/B and alphaT (lights_physical_fragment).
  let anisotropy_t = T * anisotropy_dir.x + B * anisotropy_dir.y;
  let anisotropy_b = B * anisotropy_dir.x - T * anisotropy_dir.y;
  let anisotropy_alpha_t = mix(roughness * roughness, 1.0, anisotropy_strength * anisotropy_strength);
  let has_anisotropy = anisotropy_strength > 0.0001;
  let has_clearcoat = use_specular && clearcoat > 0.0;
  let has_sheen = use_specular && max(max(sheen_color.r, sheen_color.g), sheen_color.b) > 0.0;

  // Three.js ReflectedLight terms plus the clearcoat and sheen accumulators.
  var direct_diffuse = vec3<f32>(0.0);
  var direct_specular = vec3<f32>(0.0);
  var indirect_diffuse = vec3<f32>(0.0);
  var indirect_specular = vec3<f32>(0.0);
  var clearcoat_specular_direct = vec3<f32>(0.0);
  var clearcoat_specular_indirect = vec3<f32>(0.0);
  var sheen_specular_direct = vec3<f32>(0.0);
  var sheen_specular_indirect = vec3<f32>(0.0);

  // Only MeshStandardMaterial/MeshPhysicalMaterial use the PMREM path, as in Three.js.
  let has_ibl = uniforms.normal_map_params.w > 0.5 && use_specular;
  let has_light_probe = uniforms.light_probe_params.x > 0.5;
  // Three.js r155+ adds ambient, light-probe, hemisphere, and light-map irradiance and
  // applies it as irradiance * BRDF_Lambert(diffuseColor), so it is divided by PI.
  // Standard/Physical diffuse color excludes the metallic part.
  let diffuse_color = select(albedo, albedo * (1.0 - metallic), use_specular);
  var indirect_irradiance = scene_ambient_irradiance() + light_map_irradiance;
  if has_light_probe {
    indirect_irradiance += light_probe_irradiance(N);
  }
  // A scene environment is a light source for the fallback check, also for materials it does not light.
  let scene_has_environment = uniforms.env_params.y > 0.5;

  if uniforms.num_lights == 0u && !scene_has_environment && !has_light_probe && !has_light_map && !has_scene_ambient_light() {
    // No light source at all, not even baked light: render with a basic hemispherical fallback.
    let sky_factor = 0.5 + 0.5 * N.y;
    let fallback_ambient = mix(vec3<f32>(0.1, 0.1, 0.12), vec3<f32>(0.4, 0.45, 0.5), sky_factor);
    indirect_diffuse = albedo * fallback_ambient * ao;
  } else {
    // Direct lighting from scene lights
    for (var i = 0u; i < uniforms.num_lights && i < MAX_LIGHTS; i = i + 1u) {
      let light = uniforms.lights[i];

      if light.light_type == 3u {
        indirect_irradiance += hemisphere_light_irradiance(light, N);
        continue;
      }

      var L: vec3<f32>;
      var attenuation: f32 = 1.0;

      if light.light_type == 0u {
        // Directional
        L = normalize(-light.direction.xyz);
        attenuation *= sample_shadow_for_light(i, input.world_pos, N);
      } else if light.light_type == 4u {
        // RectAreaLight approximation: finite one-sided area emitter from the
        // light center. This is intentionally cheaper than the Three.js LTC path.
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
        attenuation *= sample_shadow_for_light(i, input.world_pos, N);
      }

      let n_dot_l = saturate(dot(N, L));
      let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;
      let irradiance = radiance * n_dot_l;

      if use_specular {
        // Three.js RE_Direct_Physical.
        if has_clearcoat {
          let cc_irradiance = saturate(dot(Ncc, L)) * radiance;
          clearcoat_specular_direct += cc_irradiance * brdf_ggx(L, V, Ncc, vec3<f32>(0.04), 1.0, clearcoat_roughness);
        }
        if has_sheen {
          sheen_specular_direct += irradiance * brdf_sheen(L, V, N, sheen_color, sheen_roughness);
        }
        if has_anisotropy {
          direct_specular += irradiance * brdf_ggx_anisotropic(L, V, N, f0, specular_f90, roughness, anisotropy_alpha_t, anisotropy_t, anisotropy_b);
        } else {
          direct_specular += irradiance * brdf_ggx(L, V, N, f0, specular_f90, roughness);
        }
        direct_diffuse += irradiance * diffuse_color * RECIPROCAL_PI;
      } else if use_phong {
        // Three.js RE_Direct_BlinnPhong.
        let H = normalize(V + L);
        let n_dot_h = saturate(dot(N, H));
        let h_dot_v = saturate(dot(H, V));
        let phong_f = f_schlick(phong_specular_color, 1.0, h_dot_v);
        let phong_d = RECIPROCAL_PI * (phong_shininess * 0.5 + 1.0) * pow(n_dot_h, phong_shininess);
        direct_diffuse += irradiance * diffuse_color * RECIPROCAL_PI;
        direct_specular += irradiance * phong_f * (0.25 * phong_d) * phong_specular_strength;
      } else if use_toon {
        // MeshToonMaterial: gradientMap samples the red ramp channel at dot(N, L) * 0.5 + 0.5.
        let toon_coord = dot(N, L) * 0.5 + 0.5;
        var toon_irradiance: f32;
        if uniforms.light_probe_params.y > 0.5 {
          toon_irradiance = decode_toon_gradient_map_sample(textureSample(t_physical_sheen, s_physical_sheen_map, vec2<f32>(toon_coord, 0.0))).r;
        } else {
          let toon_width = fwidth(toon_coord) * 0.5;
          toon_irradiance = mix(0.7, 1.0, smoothstep(0.7 - toon_width, 0.7 + toon_width, toon_coord));
        }
        direct_diffuse += toon_irradiance * radiance * diffuse_color * RECIPROCAL_PI;
      } else {
        // MeshLambertMaterial: diffuse-only
        direct_diffuse += irradiance * diffuse_color * RECIPROCAL_PI;
      }
    }

    // Three.js lights_fragment_maps and RE_IndirectSpecular_Physical with a PMREM (CubeUV) envMap.
    if has_ibl {
      let ibl_irradiance_value = ibl_irradiance(N);
      var radiance: vec3<f32>;
      if has_anisotropy {
        radiance = ibl_anisotropy_radiance(V, N, roughness, anisotropy_b, anisotropy_strength);
      } else {
        radiance = ibl_radiance(V, N, roughness);
      }
      if has_clearcoat {
        let clearcoat_radiance = ibl_radiance(V, Ncc, clearcoat_roughness);
        clearcoat_specular_indirect += clearcoat_radiance * environment_brdf(Ncc, V, vec3<f32>(0.04), 1.0, clearcoat_roughness);
      }
      if has_sheen {
        sheen_specular_indirect += ibl_irradiance_value * sheen_color * ibl_sheen_brdf(N, V, sheen_roughness);
      }
      let scattering = compute_multiscattering(N, V, f0, specular_f90, roughness);
      let total_scattering = scattering.single + scattering.multi;
      let cosine_weighted_irradiance = ibl_irradiance_value * RECIPROCAL_PI;
      let ibl_diffuse = diffuse_color * (1.0 - max(max(total_scattering.r, total_scattering.g), total_scattering.b));
      indirect_specular += radiance * scattering.single + scattering.multi * cosine_weighted_irradiance;
      indirect_diffuse += ibl_diffuse * cosine_weighted_irradiance;
    }
    // Three.js RE_IndirectDiffuse: irradiance * BRDF_Lambert(diffuseColor).
    indirect_diffuse += indirect_irradiance * diffuse_color * RECIPROCAL_PI;

    // Three.js aomap_fragment.
    indirect_diffuse *= ao;
    clearcoat_specular_indirect *= ao;
    sheen_specular_indirect *= ao;
    if has_ibl {
      indirect_specular *= compute_specular_occlusion(n_dot_v, ao, roughness);
    }
  }

  var total_diffuse = direct_diffuse + indirect_diffuse;
  let total_specular = direct_specular + indirect_specular;

  if use_specular && transmission > 0.0001 {
    let dispersion = max(uniforms.attenuation_color.w, 0.0);
    let refracted_dir = refract(-V, N, 1.0 / ior);
    let transmittance = volume_attenuation(thickness, uniforms.attenuation_color.rgb, attenuation_distance);
    let scene_offset = refracted_dir.xy * thickness * 0.04;
    let scene_uv = clamp(screen_uv + scene_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let transmitted_sample = sample_transmission_scene_color(scene_uv, roughness, ior);
    var transmitted_light = transmitted_sample * transmittance;
    if dispersion > 0.0001 {
      let half_spread = max(ior - 1.0, 0.0) * 0.025 * dispersion;
      let ior_r = clamp(ior - half_spread, 1.0, 2.333);
      let ior_b = clamp(ior + half_spread, 1.0, 2.333);
      let refracted_r = refract(-V, N, 1.0 / ior_r);
      let refracted_b = refract(-V, N, 1.0 / ior_b);
      let uv_r = clamp(screen_uv + refracted_r.xy * thickness * 0.04, vec2<f32>(0.0), vec2<f32>(1.0));
      let uv_b = clamp(screen_uv + refracted_b.xy * thickness * 0.04, vec2<f32>(0.0), vec2<f32>(1.0));
      transmitted_light = vec3<f32>(
        sample_transmission_scene_color(uv_r, roughness, ior_r).r,
        transmitted_sample.g,
        sample_transmission_scene_color(uv_b, roughness, ior_b).b,
      ) * transmittance;
    }
    if has_ibl {
      let environment_refraction = texture_cube_uv(rotate_environment_direction(refracted_dir), roughness) * uniforms.ibl_params.x * transmittance;
      transmitted_light = mix(transmitted_light, environment_refraction, 0.35);
    }
    // Three.js transmission_fragment replaces only the diffuse part.
    total_diffuse = mix(total_diffuse, transmitted_light, transmission);
  }

  // Emissive
  let emissive_sample = decode_emissive_map_sample(textureSample(t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2))).rgb;
  var lo = total_diffuse + total_specular + uniforms.emissive.rgb * emissive_sample;

  if has_sheen {
    // Three.js sheen energy compensation.
    let sheen_energy_comp = 1.0 - 0.157 * max(max(sheen_color.r, sheen_color.g), sheen_color.b);
    lo = lo * sheen_energy_comp + sheen_specular_direct + sheen_specular_indirect;
  }
  if has_clearcoat {
    let dot_nv_cc = saturate(dot(Ncc, V));
    let fcc = f_schlick(vec3<f32>(0.04), 1.0, dot_nv_cc);
    lo = lo * (vec3<f32>(1.0) - clearcoat * fcc) + (clearcoat_specular_direct + clearcoat_specular_indirect) * clearcoat;
  }

  if legacy_material_env {
    // Three.js envmap_fragment for MeshLambertMaterial and MeshPhongMaterial.
    let combine = u32(uniforms.env_map_params.x + 0.5);
    let legacy_env_mode = u32(uniforms.env_map_params.z + 0.5);
    var legacy_env_dir = reflect(-V, N);
    if legacy_env_mode == 2u {
      legacy_env_dir = refract(-V, N, uniforms.env_map_params.w);
    }
    let legacy_env_color = legacy_environment_color(legacy_env_dir);
    let legacy_strength = legacy_env_reflectivity * select(1.0, phong_specular_strength, use_phong);
    if combine == 2u {
      lo = lo + legacy_env_color * legacy_strength;
    } else if combine == 1u {
      lo = mix(lo, legacy_env_color, legacy_strength);
    } else {
      lo = mix(lo, lo * legacy_env_color, legacy_strength);
    }
  }

  // Tone mapping and output color conversion, as the Three.js tonemapping and colorspace fragments.
  let mapped = apply_material_tone_mapping(lo);
  let output_mapped = apply_output_color_space(mapped);
  let fogged = apply_fog(output_mapped, fog_depth(input.world_pos));

  return output_color(fogged, alpha);
}

fn apply_material_tone_mapping(color: vec3<f32>) -> vec3<f32> {
  return tone_map(color, uniforms.output_params.y, uniforms.output_params.w);
}

// Three.js sRGBTransferOETF (colorspace_pars_fragment): the piecewise sRGB curve, not a 2.2 gamma.
fn srgb_transfer_oetf(color: vec3<f32>) -> vec3<f32> {
  let encoded = pow(max(color, vec3<f32>(0.0)), vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
  return select(encoded, color * 12.92, color <= vec3<f32>(0.0031308));
}

fn apply_output_color_space(color: vec3<f32>) -> vec3<f32> {
  if uniforms.output_params.x > 0.5 {
    return color;
  }
  return srgb_transfer_oetf(color);
}
