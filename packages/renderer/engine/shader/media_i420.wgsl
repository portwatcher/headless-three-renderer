@group(0) @binding(0) var rgba_source: texture_2d<f32>;
@group(0) @binding(1) var y_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var u_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(3) var v_plane: texture_storage_2d<r8unorm, write>;

fn limited_y(rgb: vec3<f32>) -> f32 {
  return (16.0 + 219.0 * dot(rgb, vec3<f32>(0.299, 0.587, 0.114))) / 255.0;
}

fn limited_uv(rgb: vec3<f32>) -> vec2<f32> {
  let cb = dot(rgb, vec3<f32>(-0.168736, -0.331264, 0.5));
  let cr = dot(rgb, vec3<f32>(0.5, -0.418688, -0.081312));
  return (vec2<f32>(128.0) + 224.0 * vec2<f32>(cb, cr)) / 255.0;
}

@compute @workgroup_size(8, 8)
fn convert(@builtin(global_invocation_id) id: vec3<u32>) {
  let chroma_size = textureDimensions(u_plane);
  if (id.x >= chroma_size.x || id.y >= chroma_size.y) {
    return;
  }
  let base = vec2<i32>(id.xy * 2u);
  let c0 = textureLoad(rgba_source, base, 0).rgb;
  let c1 = textureLoad(rgba_source, base + vec2<i32>(1, 0), 0).rgb;
  let c2 = textureLoad(rgba_source, base + vec2<i32>(0, 1), 0).rgb;
  let c3 = textureLoad(rgba_source, base + vec2<i32>(1, 1), 0).rgb;
  textureStore(y_plane, base, vec4<f32>(limited_y(c0), 0.0, 0.0, 1.0));
  textureStore(y_plane, base + vec2<i32>(1, 0), vec4<f32>(limited_y(c1), 0.0, 0.0, 1.0));
  textureStore(y_plane, base + vec2<i32>(0, 1), vec4<f32>(limited_y(c2), 0.0, 0.0, 1.0));
  textureStore(y_plane, base + vec2<i32>(1, 1), vec4<f32>(limited_y(c3), 0.0, 0.0, 1.0));
  let uv = limited_uv((c0 + c1 + c2 + c3) * 0.25);
  textureStore(u_plane, vec2<i32>(id.xy), vec4<f32>(uv.x, 0.0, 0.0, 1.0));
  textureStore(v_plane, vec2<i32>(id.xy), vec4<f32>(uv.y, 0.0, 0.0, 1.0));
}
