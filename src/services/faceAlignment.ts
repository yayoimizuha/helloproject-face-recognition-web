import type { FaceDetection } from '../types'
import type { FaceLandmarks } from './MediaPipeLandmarker'

const MARGIN_RATIO = 0.15

/**
 * 顔が小さすぎるか判定する。
 * @param minPx 短辺の最小ピクセル数（デフォルト 48px）
 */
export function isFaceTooSmall(detection: FaceDetection, minPx = 48): boolean {
  const { width, height } = detection.bbox
  return Math.min(width, height) < minPx
}

/**
 * 両目座標から顔の傾き角度（ラジアン）を計算する。
 * eyeKeypoints の座標は cropFaceRaw で切り出したキャンバス上の正規化座標 (0〜1)。
 */
function calcRollAngle(eyes: Pick<FaceLandmarks, 'rightEye' | 'leftEye'>): number {
  const dx = eyes.leftEye.x - eyes.rightEye.x
  const dy = eyes.leftEye.y - eyes.rightEye.y
  return Math.atan2(dy, dx)  // ラジアン、水平なら 0
}

/**
 * bbox をマージン付きでクロップした生の顔 Canvas を返す（回転なし）。
 * MediaPipe ランドマーク取得の入力として使う。
 */
export function cropFaceRaw(
  source: CanvasImageSource,
  detection: FaceDetection,
  size: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, size, size)

  const { x, y, width, height } = detection.bbox
  const mx = width * MARGIN_RATIO
  const my = height * MARGIN_RATIO
  const sx = Math.max(0, x - mx)
  const sy = Math.max(0, y - my)
  const sw = width + 2 * mx
  const sh = height + 2 * my

  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, size, size)
  return canvas
}

/**
 * 両目座標を元に顔を回転補正してクロップした Canvas を返す。
 *
 * 処理:
 *   1. bbox + マージンで粗くクロップ
 *   2. 両目の角度から回転量を算出
 *   3. Canvas の中心を軸にアフィン回転
 *   4. inputSize × inputSize にリサイズ
 *
 * eyes が null（ランドマーク取得失敗）の場合は回転なしの cropFaceRaw を返す。
 */
export function cropFace(
  source: CanvasImageSource,
  detection: FaceDetection,
  inputSize: number,
  eyes: Pick<FaceLandmarks, 'rightEye' | 'leftEye'> | null,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = inputSize
  canvas.height = inputSize
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, inputSize, inputSize)

  const { x, y, width, height } = detection.bbox
  const mx = width * MARGIN_RATIO
  const my = height * MARGIN_RATIO
  const sx = Math.max(0, x - mx)
  const sy = Math.max(0, y - my)
  const sw = width + 2 * mx
  const sh = height + 2 * my

  if (!eyes) {
    // ランドマーク取得失敗 → 補正なしで描画
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, inputSize, inputSize)
    return canvas
  }

  const angle = calcRollAngle(eyes)

  // キャンバス中心を軸に回転してから描画
  const cx = inputSize / 2
  const cy = inputSize / 2
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(-angle)   // 傾きを打ち消す方向に回転
  ctx.translate(-cx, -cy)
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, inputSize, inputSize)
  ctx.restore()

  return canvas
}
