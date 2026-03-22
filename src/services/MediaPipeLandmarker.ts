/**
 * MediaPipe FaceDetector を使って bbox 内の顔に対し
 * BlazeFace short-range の6点ランドマークを取得するサービス。
 *
 * YOLO-Face で bbox 検出済みの顔画像クロップに対して呼び出し、
 * 両目座標から傾き角度を返す。
 */
import {
  FaceDetector,
  FilesetResolver,
  type Detection,
} from '@mediapipe/tasks-vision'

// BlazeFace short-range キーポイントインデックス（実測6点）
// 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear tragion, 5: left ear tragion
const KP_RIGHT_EYE = 0
const KP_LEFT_EYE  = 1
const KP_NOSE_TIP  = 2
const KP_MOUTH     = 3
const KP_RIGHT_EAR = 4
const KP_LEFT_EAR  = 5

export interface FaceLandmarks {
  /** 正規化座標 (0〜1) — クロップキャンバス上 */
  rightEye: { x: number; y: number }
  leftEye:  { x: number; y: number }
  noseTip:  { x: number; y: number }
  mouth:    { x: number; y: number }
  rightEar: { x: number; y: number }
  leftEar:  { x: number; y: number }
}

/** 後方互換 */
export type EyeKeypoints = Pick<FaceLandmarks, 'rightEye' | 'leftEye'>

export class MediaPipeLandmarker {
  private detector: FaceDetector | null = null

  async initialize(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
    )
    this.detector = await FaceDetector.createFromModelPath(
      vision,
      '/models/blaze_face_short_range.tflite',
    )
  }

  /**
   * クロップ済み顔 Canvas から6点ランドマークを取得する。
   * 検出できなければ null を返す。
   */
  detectLandmarks(canvas: HTMLCanvasElement): FaceLandmarks | null {
    if (!this.detector) return null

    let detections: Detection[]
    try {
      const result = this.detector.detect(canvas)
      detections = result.detections
    } catch {
      return null
    }

    if (detections.length === 0) return null

    // スコア最大の検出を使用
    const best = detections.reduce((a, b) =>
      (a.categories[0]?.score ?? 0) >= (b.categories[0]?.score ?? 0) ? a : b
    )

    const kps = best.keypoints
    if (!kps || kps.length < 6) return null

    return {
      rightEye: { x: kps[KP_RIGHT_EYE].x, y: kps[KP_RIGHT_EYE].y },
      leftEye:  { x: kps[KP_LEFT_EYE].x,  y: kps[KP_LEFT_EYE].y },
      noseTip:  { x: kps[KP_NOSE_TIP].x,  y: kps[KP_NOSE_TIP].y },
      mouth:    { x: kps[KP_MOUTH].x,     y: kps[KP_MOUTH].y },
      rightEar: { x: kps[KP_RIGHT_EAR].x, y: kps[KP_RIGHT_EAR].y },
      leftEar:  { x: kps[KP_LEFT_EAR].x,  y: kps[KP_LEFT_EAR].y },
    }
  }

  /** @deprecated detectLandmarks() を使うこと */
  detectEyes(canvas: HTMLCanvasElement): EyeKeypoints | null {
    return this.detectLandmarks(canvas)
  }

  dispose(): void {
    this.detector?.close()
    this.detector = null
  }
}
