import type * as ort from 'onnxruntime-web'

// ── 顔検出 ──────────────────────────────────────────────────────────────────

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export type LandmarkLabel = 'rightEye' | 'leftEye' | 'noseTip' | 'mouthRight' | 'mouthLeft'

export interface Keypoint {
  x: number
  y: number
  label?: LandmarkLabel
}

/** 顔検出アダプターが返す検出結果（ライブラリ非依存、ピクセル座標） */
export interface FaceDetection {
  bbox: BoundingBox
  keypoints: Keypoint[]   // 5点ランドマーク
  score: number           // 検出スコア 0〜1
}

// ── 顔検出アダプター ────────────────────────────────────────────────────────

export interface DetectorConfig {
  modelPath: string
  wasmPath?: string
  minDetectionScore?: number
}

export interface FaceDetectorAdapter {
  initialize(config: DetectorConfig): Promise<void>
  detect(source: CanvasImageSource, timestamp?: number): Promise<FaceDetection[]>
  /** IMAGE ↔ VIDEO モードを動的に切り替える */
  setRunningMode(mode: 'IMAGE' | 'VIDEO'): Promise<void>
  dispose(): void
}

// ── モデル設定 ───────────────────────────────────────────────────────────────

export interface ModelConfig {
  session: ort.InferenceSession
  classNames: string[]
  mean: number[]
  std: number[]
  inputSize: number
  /** モデルが anomaly_score 出力を持つか */
  hasAnomalyOutput: boolean
  /**
   * AnomalyClassifier の閾値（sigmoid 出力、0〜1）。
   * anomaly_score > anomalyThreshold のとき異常判定。
   * ONNX メタデータの anomaly_threshold キーから取得。
   */
  anomalyThreshold: number | null
}

// ── 推論結果 ─────────────────────────────────────────────────────────────────

/** identity=null のときの理由 */
export type UnknownReason =
  | 'too_small'        // bbox が小さすぎて推論スキップ
  | 'anomaly'          // anomaly_score > threshold
  | 'low_confidence'   // 信頼度が認識閾値未満
  | 'no_anomaly_model' // モデルに anomaly 出力がなく isAnomaly 判定不可

export interface ClassificationResult {
  predClassIdx: number
  identity: string | null     // null = Unknown
  confidence: number          // softmax max prob 0〜1
  anomalyScore: number | null
  isAnomaly: boolean
  /** identity=null のときの理由 */
  unknownReason?: UnknownReason
  /** @deprecated unknownReason === 'too_small' で代替 */
  tooSmall?: boolean
}

/**
 * 元画像ピクセル座標での BlazeFace 6点ランドマーク
 * 0:rightEye 1:leftEye 2:noseTip 3:mouth 4:rightEar 5:leftEar
 */
export interface FaceLandmarksPixel {
  rightEye: { x: number; y: number }
  leftEye:  { x: number; y: number }
  noseTip:  { x: number; y: number }
  mouth:    { x: number; y: number }
  rightEar: { x: number; y: number }
  leftEar:  { x: number; y: number }
}

export interface RecognitionResult {
  detection: FaceDetection
  classification: ClassificationResult
  /** MediaPipe で取得した5点ランドマーク（元画像ピクセル座標）。取得失敗時は null */
  landmarks: FaceLandmarksPixel | null
}

// ── パイプラインバッファ ─────────────────────────────────────────────────────

export interface FrameEntry {
  bitmap: ImageBitmap
  timestamp: DOMHighResTimeStamp
}

export interface DetectionEntry {
  bitmap: ImageBitmap
  detections: FaceDetection[]
  timestamp: DOMHighResTimeStamp
}

export interface ResultStore {
  results: RecognitionResult[]
  timestamp: DOMHighResTimeStamp
}

export interface PipelineConfig {
  frameQueueCapacity: number
  detectionQueueCapacity: number
  detectIntervalMs: number
  recognizeIntervalMs: number
  maxBatchSize: number
  resultTtlMs: number
}

// ── バッチ推論 ───────────────────────────────────────────────────────────────

export interface BatchMeta {
  entryTimestamp: DOMHighResTimeStamp
  detection: FaceDetection
}
