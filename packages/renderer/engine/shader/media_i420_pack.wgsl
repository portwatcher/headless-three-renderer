@group(0) @binding(0) var y_plane: texture_2d<f32>;
@group(0) @binding(1) var u_plane: texture_2d<f32>;
@group(0) @binding(2) var v_plane: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> packed_i420: array<u32>;

fn byte_at(index: u32) -> u32 {
  let y_size = textureDimensions(y_plane);
  let chroma_size = textureDimensions(u_plane);
  let y_length = y_size.x * y_size.y;
  let chroma_length = chroma_size.x * chroma_size.y;
  if (index < y_length) {
    let position = vec2<i32>(i32(index % y_size.x), i32(index / y_size.x));
    return u32(round(clamp(textureLoad(y_plane, position, 0).r, 0.0, 1.0) * 255.0));
  }
  if (index < y_length + chroma_length) {
    let offset = index - y_length;
    let position = vec2<i32>(i32(offset % chroma_size.x), i32(offset / chroma_size.x));
    return u32(round(clamp(textureLoad(u_plane, position, 0).r, 0.0, 1.0) * 255.0));
  }
  if (index < y_length + 2u * chroma_length) {
    let offset = index - y_length - chroma_length;
    let position = vec2<i32>(i32(offset % chroma_size.x), i32(offset / chroma_size.x));
    return u32(round(clamp(textureLoad(v_plane, position, 0).r, 0.0, 1.0) * 255.0));
  }
  return 0u;
}

@compute @workgroup_size(64)
fn convert(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
) {
  let word_index = id.x + id.y * groups.x * 64u;
  if (word_index >= arrayLength(&packed_i420)) {
    return;
  }
  let base = word_index * 4u;
  packed_i420[word_index] = byte_at(base)
    | (byte_at(base + 1u) << 8u)
    | (byte_at(base + 2u) << 16u)
    | (byte_at(base + 3u) << 24u);
}
