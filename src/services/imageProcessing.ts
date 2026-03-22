/**
 * Canvas から ImageNet 正規化済み CHW Float32Array を生成する。
 * shape: [1, 3, H, W] の NCHW テンソル用バッファを返す。
 */
export function canvasToTensor(
  canvas: HTMLCanvasElement,
  mean: number[],
  std: number[]
): Float32Array {
  const H = canvas.height
  const W = canvas.width
  const ctx = canvas.getContext('2d')!
  const { data } = ctx.getImageData(0, 0, W, H)  // RGBA Uint8ClampedArray

  const tensor = new Float32Array(3 * H * W)
  const HW = H * W

  for (let i = 0; i < HW; i++) {
    // (pixel / 255 - mean) / std
    tensor[0 * HW + i] = (data[i * 4 + 0] / 255 - mean[0]) / std[0]  // R
    tensor[1 * HW + i] = (data[i * 4 + 1] / 255 - mean[1]) / std[1]  // G
    tensor[2 * HW + i] = (data[i * 4 + 2] / 255 - mean[2]) / std[2]  // B
  }
  return tensor
}

/**
 * 複数の [1,3,H,W] テンソルを [B,3,H,W] にバッチ結合する
 */
export function stackTensors(tensors: Float32Array[], H: number, W: number): Float32Array {
  const B = tensors.length
  const CHW = 3 * H * W
  const out = new Float32Array(B * CHW)
  for (let i = 0; i < B; i++) {
    out.set(tensors[i], i * CHW)
  }
  return out
}
