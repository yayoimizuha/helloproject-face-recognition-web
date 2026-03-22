import * as ort from 'onnxruntime-web/all'
import type { FaceDetection } from '../../types'

export type YoloModelSize = 'n' | 's' | 'm' | 'l'
export type ExecutionProvider = 'wasm' | 'webgpu'

const MODEL_INPUT_SIZE = 640

export interface YoloDetectorConfig {
  modelSize?: YoloModelSize
  minDetectionScore?: number
  ep?: ExecutionProvider
}

/**
 * YOLOv12-Face detector using onnxruntime-web.
 * Output format: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id] (xyxy, post-NMS)
 * Coordinates are in the 640x640 input space.
 */
export class YoloFaceDetector {
  private session: ort.InferenceSession | null = null
  private minScore: number
  private modelSize: YoloModelSize
  private ep: ExecutionProvider
  private modelInputSize = MODEL_INPUT_SIZE

  constructor(config: YoloDetectorConfig = {}) {
    this.minScore = config.minDetectionScore ?? 0.4
    this.modelSize = config.modelSize ?? 'n'
    this.ep = config.ep ?? 'wasm'
  }

  get modelUrl(): string {
    return `/models/yolov12${this.modelSize}-face.onnx`
  }

  async initialize(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: [this.ep],
      graphOptimizationLevel: 'all',
    })
  }

  async detect(
    source: CanvasImageSource,
    minScore?: number,
  ): Promise<FaceDetection[]> {
    if (!this.session) return []

    const threshold = minScore ?? this.minScore

    // source のサイズを取得
    let srcW: number
    let srcH: number
    if (source instanceof HTMLVideoElement) {
      srcW = source.videoWidth
      srcH = source.videoHeight
    } else if (source instanceof HTMLImageElement) {
      srcW = source.naturalWidth
      srcH = source.naturalHeight
    } else if (source instanceof ImageBitmap) {
      srcW = source.width
      srcH = source.height
    } else {
      srcW = (source as HTMLCanvasElement).width
      srcH = (source as HTMLCanvasElement).height
    }

    // letterbox リサイズ: 640x640 に収まるようアスペクト維持でパディング
    const scale = Math.min(this.modelInputSize / srcW, this.modelInputSize / srcH)
    const scaledW = Math.round(srcW * scale)
    const scaledH = Math.round(srcH * scale)
    const padX = Math.floor((this.modelInputSize - scaledW) / 2)
    const padY = Math.floor((this.modelInputSize - scaledH) / 2)

    // Canvas に letterbox 描画
    const canvas = document.createElement('canvas')
    canvas.width = this.modelInputSize
    canvas.height = this.modelInputSize
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#808080'
    ctx.fillRect(0, 0, this.modelInputSize, this.modelInputSize)
    ctx.drawImage(source, padX, padY, scaledW, scaledH)

    // ImageData -> Float32 tensor [1, 3, H, W], normalized 0-1
    const imageData = ctx.getImageData(0, 0, this.modelInputSize, this.modelInputSize)
    const input = imageDataToTensor(imageData, this.modelInputSize)

    const feeds = { images: input }
    const results = await this.session.run(feeds)
    const output = results['output0'].data as Float32Array
    // output: [1, 300, 6] flattened -> length = 300 * 6
    const numBoxes = 300
    const stride = 6

    const detections: FaceDetection[] = []
    for (let i = 0; i < numBoxes; i++) {
      const base = i * stride
      const score = output[base + 4]
      if (score < threshold) continue

      // coordinates in 640x640 input space, convert back to source space
      const x1Model = output[base + 0]
      const y1Model = output[base + 1]
      const x2Model = output[base + 2]
      const y2Model = output[base + 3]

      // undo letterbox: remove padding, undo scale
      const x1Src = (x1Model - padX) / scale
      const y1Src = (y1Model - padY) / scale
      const x2Src = (x2Model - padX) / scale
      const y2Src = (y2Model - padY) / scale

      // clamp to source dimensions
      const x1 = Math.max(0, x1Src)
      const y1 = Math.max(0, y1Src)
      const x2 = Math.min(srcW, x2Src)
      const y2 = Math.min(srcH, y2Src)

      const w = x2 - x1
      const h = y2 - y1
      if (w <= 0 || h <= 0) continue

      detections.push({
        bbox: { x: x1, y: y1, width: w, height: h },
        keypoints: [],
        score,
      })
    }

    return detections
  }

  dispose(): void {
    this.session?.release()
    this.session = null
  }
}

/** ImageData を NCHW Float32 テンソル (0-1 正規化) に変換 */
function imageDataToTensor(
  imageData: ImageData,
  size: number,
): ort.Tensor {
  const { data } = imageData
  const n = size * size
  const float32 = new Float32Array(3 * n)
  for (let i = 0; i < n; i++) {
    float32[i]         = data[i * 4 + 0] / 255.0  // R channel
    float32[n + i]     = data[i * 4 + 1] / 255.0  // G channel
    float32[2 * n + i] = data[i * 4 + 2] / 255.0  // B channel
  }
  return new ort.Tensor('float32', float32, [1, 3, size, size])
}
