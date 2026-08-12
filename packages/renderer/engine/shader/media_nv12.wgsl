@group(0) @binding(0) var rgba_source: texture_2d<f32>;
@group(0) @binding(1) var y_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_plane: texture_storage_2d<rg8unorm, write>;

fn limited_y(rgb: vec3<f32>) -> f32 {
  return (16.0 + 219.0 * dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722))) / 255.0;
}

fn limited_uv(rgb: vec3<f32>) -> vec2<f32> {
  let cb = dot(rgb, vec3<f32>(-0.114572, -0.385428, 0.5));
  let cr = dot(rgb, vec3<f32>(0.5, -0.454153, -0.045847));
  return (vec2<f32>(128.0) + 224.0 * vec2<f32>(cb, cr)) / 255.0;
}

@compute @workgroup_size(8, 8)
fn convert(@builtin(global_invocation_id) id: vec3<u32>) {
  let uv_size = textureDimensions(uv_plane);
  if (id.x >= uv_size.x || id.y >= uv_size.y) {
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
  textureStore(uv_plane, vec2<i32>(id.xy), vec4<f32>(limited_uv((c0 + c1 + c2 + c3) * 0.25), 0.0, 1.0));
}
