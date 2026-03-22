import type { RecognitionResult, FaceDetection, FaceLandmarksPixel } from '../types'

const COLORS = {
  success: '#22c55e',
  warning: '#f97316',
  error: '#ef4444',
  detect: '#3b82f6',
}

// ランドマーク点の色（半透明で塗る）
const LM_COLORS: Record<keyof FaceLandmarksPixel, string> = {
  rightEye: '#38bdf8', // 水色
  leftEye:  '#38bdf8', // 水色
  noseTip:  '#facc15', // 黄
  mouth:    '#f472b6', // ピンク
  rightEar: '#a78bfa', // 紫
  leftEar:  '#a78bfa', // 紫
}

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  results: RecognitionResult[],
  detections: FaceDetection[],
  width: number,
  height: number,
  scaleX = 1,
  scaleY = 1,
  showLandmarks = true,
): void {
  ctx.clearRect(0, 0, width, height)

  const resultMap = new Map<FaceDetection, RecognitionResult>()
  for (const r of results) resultMap.set(r.detection, r)

  for (const r of results) {
    const { detection: d, classification: c, landmarks } = r
    const color = c.identity ? COLORS.success
      : c.unknownReason === 'anomaly' ? COLORS.error
      : COLORS.warning
    drawBox(ctx, d, color, scaleX, scaleY)

    let label: string
    if (c.identity) {
      label = `${c.identity} ${(c.confidence * 100).toFixed(1)}%`
    } else {
      switch (c.unknownReason) {
        case 'too_small':
          label = 'Too Small'
          break
        case 'anomaly':
          label = `Anomaly (score=${c.anomalyScore?.toFixed(3) ?? '?'})`
          break
        case 'low_confidence':
          label = `Low Conf (${(c.confidence * 100).toFixed(1)}%)`
          break
        case 'no_anomaly_model':
          label = `Unknown (${(c.confidence * 100).toFixed(1)}%)`
          break
        default:
          label = 'Unknown'
      }
    }
    drawLabel(ctx, d, label, color, scaleX, scaleY)
    drawKeypoints(ctx, d, color, scaleX, scaleY)
    if (showLandmarks && landmarks) {
      drawLandmarks(ctx, landmarks, scaleX, scaleY)
    }
  }

  for (const d of detections) {
    if (!resultMap.has(d)) {
      drawBox(ctx, d, COLORS.detect, scaleX, scaleY)
    }
  }
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  d: FaceDetection,
  color: string,
  sx: number,
  sy: number,
): void {
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(d.bbox.x * sx, d.bbox.y * sy, d.bbox.width * sx, d.bbox.height * sy)
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  d: FaceDetection,
  text: string,
  color: string,
  sx: number,
  sy: number,
): void {
  const LABEL_H = 18
  const PADDING = 4
  const FONT = 'bold 13px system-ui'
  ctx.font = FONT

  const bboxTop = d.bbox.y * sy
  const bboxLeft = d.bbox.x * sx

  // bbox の上にラベルが収まるか判定（上端から LABEL_H + 2px 必要）
  const aboveY = bboxTop - 2
  const placeAbove = aboveY >= LABEL_H

  const bgY = placeAbove ? aboveY - LABEL_H : bboxTop + 2
  const textY = bgY + LABEL_H - PADDING

  const metrics = ctx.measureText(text)
  ctx.fillStyle = color + 'cc'
  ctx.fillRect(bboxLeft, bgY, metrics.width + PADDING * 2, LABEL_H)
  ctx.fillStyle = '#fff'
  ctx.fillText(text, bboxLeft + PADDING, textY)
}

function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  d: FaceDetection,
  color: string,
  sx: number,
  sy: number,
): void {
  ctx.fillStyle = color
  for (const kp of d.keypoints) {
    ctx.beginPath()
    ctx.arc(kp.x * sx, kp.y * sy, 3, 0, Math.PI * 2)
    ctx.fill()
  }
}

/**
 * MediaPipe 5点ランドマークを半透明の色付き円で描画する。
 * 塗り: 70% 不透明 / 輪郭: 白 30%
 */
function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  lm: FaceLandmarksPixel,
  sx: number,
  sy: number,
): void {
  const RADIUS = 5

  for (const key of Object.keys(lm) as (keyof FaceLandmarksPixel)[]) {
    const { x, y } = lm[key]
    const color = LM_COLORS[key]

    // 塗りつぶし（半透明）
    ctx.beginPath()
    ctx.arc(x * sx, y * sy, RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = color + 'b3'   // alpha ≈ 0.70
    ctx.fill()

    // 輪郭（白・薄め）
    ctx.beginPath()
    ctx.arc(x * sx, y * sy, RADIUS, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(255,255,255,0.5)'
    ctx.lineWidth = 1
    ctx.stroke()
  }
}
