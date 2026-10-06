// Minimal glTF quad with a VRMC_materials_mtoon face material, as the npcify AIGC generator
// writes it: shading shift texture red 0 at the edge (u < 0.5) and 1 inside (u >= 0.5).
const SHIFT_TEXTURE_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGNgYGD4/5+B4T8ACf8C/rR1zsoAAAAASUVORK5CYII='

export const mtoonFaceGltf = function () {
  const positions = new Float32Array([-0.75, -0.75, 0, 0.75, -0.75, 0, 0.75, 0.75, 0, -0.75, 0.75, 0])
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
  const uvs = new Float32Array([0, 1, 1, 1, 1, 0, 0, 0])
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3])
  const buffer = Buffer.concat([positions, normals, uvs, indices].map((array) => Buffer.from(array.buffer)))
  const view = (byteOffset, byteLength) => ({ buffer: 0, byteOffset, byteLength })
  return JSON.stringify({
    asset: { version: '2.0' },
    extensionsUsed: ['VRMC_materials_mtoon'],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }] }],
    materials: [{
      name: 'Face',
      pbrMetallicRoughness: { baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 1 },
      extensions: {
        VRMC_materials_mtoon: {
          specVersion: '1.0',
          shadeColorFactor: [0.78, 0.76, 0.84],
          shadingToonyFactor: 0.9,
          shadingShiftFactor: -0.05,
          shadingShiftTexture: { index: 0, scale: 0.75 },
          parametricRimColorFactor: [0, 0, 0],
          outlineWidthMode: 'none',
        },
      },
    }],
    textures: [{ source: 0, sampler: 0 }],
    samplers: [{ magFilter: 9728, minFilter: 9728, wrapS: 33071, wrapT: 33071 }],
    images: [{ uri: `data:image/png;base64,${SHIFT_TEXTURE_PNG}`, mimeType: 'image/png' }],
    buffers: [{ byteLength: buffer.length, uri: `data:application/octet-stream;base64,${buffer.toString('base64')}` }],
    bufferViews: [view(0, 48), view(48, 48), view(96, 32), view(128, 12)],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-0.75, -0.75, 0], max: [0.75, 0.75, 0] },
      { bufferView: 1, componentType: 5126, count: 4, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 4, type: 'VEC2' },
      { bufferView: 3, componentType: 5123, count: 6, type: 'SCALAR' },
    ],
  })
}
