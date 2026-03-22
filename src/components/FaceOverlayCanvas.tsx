import { useEffect, useRef, useCallback } from 'react'
import type { RecognitionResult, FaceDetection } from '../types'
import { drawOverlay } from '../services/overlayRenderer'

interface Props {
  results: RecognitionResult[]
  detections: FaceDetection[]
  /** overlay canvas の表示幅 */
  width: number
  /** overlay canvas の表示高さ */
  height: number
  /** 検出座標が基準とする元ソースの幅（省略時は width と同じ） */
  sourceWidth?: number
  /** 検出座標が基準とする元ソースの高さ（省略時は height と同じ） */
  sourceHeight?: number
  /** ランドマーク表示ON/OFF（デフォルト true） */
  showLandmarks?: boolean
}

export function FaceOverlayCanvas({
  results, detections, width, height, sourceWidth, sourceHeight, showLandmarks = true,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const scaleX = width / (sourceWidth ?? width)
    const scaleY = height / (sourceHeight ?? height)
    drawOverlay(ctx, results, detections, width, height, scaleX, scaleY, showLandmarks)
  }, [results, detections, width, height, sourceWidth, sourceHeight, showLandmarks])

  useEffect(() => {
    draw()
  }, [draw])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  )
}
