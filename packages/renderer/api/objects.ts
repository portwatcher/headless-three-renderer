import type { ThreeObject3DLike } from './types'

export function objectChildren(object: ThreeObject3DLike, label = 'object.children'): ThreeObject3DLike[] {
  const children = object.children
  if (children == null) return []
  if (!Array.isArray(children)) {
    throw new TypeError(`${label} must be an array.`)
  }
  return children
}

export function validateObjectChildrenTree(object: ThreeObject3DLike, label = 'object'): void {
  const children = objectChildren(object, `${label}.children`)
  for (let index = 0; index < children.length; index += 1) {
    const child = children[index]
    const childLabel = `${label}.children[${index}]`
    if (!child || typeof child !== 'object' || Array.isArray(child)) {
      throw new TypeError(`${childLabel} must be an object.`)
    }
    validateObjectChildrenTree(child, childLabel)
  }
}
