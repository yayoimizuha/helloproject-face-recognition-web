import * as ort from 'onnxruntime-web'
import type { ModelConfig, ClassificationResult, FaceDetection, RecognitionResult, FaceLandmarksPixel, UnknownReason } from '../types'
import { cropFace, isFaceTooSmall } from './faceAlignment'
import type { FaceLandmarks } from './MediaPipeLandmarker'
import { canvasToTensor, stackTensors } from './imageProcessing'

const MARGIN_RATIO = 0.15

/**
 * クロップキャンバス上の正規化座標 (0〜1) を元画像のピクセル座標に変換する。
 *
 * cropFaceRaw は (sx, sy, sw, sh) → (0, 0, size, size) に
 * アスペクト比を無視してストレッチして描画する。
 * そのため逆変換は sw/sh ではなく同じ size で割り戻す必要はなく、
 * nx * sw + sx、ny * sh + sy で戻せばよい。
 * ただし sw と sh が異なる場合（長方形 bbox）は X/Y それぞれ独立したスケールが必要。
 */
function landmarksToPixel(
  lm: FaceLandmarks,
  detection: FaceDetection,
): FaceLandmarksPixel {
  const { x, y, width, height } = detection.bbox
  const mx = width * MARGIN_RATIO
  const my = height * MARGIN_RATIO
  const sx = Math.max(0, x - mx)
  const sy = Math.max(0, y - my)
  const sw = width + 2 * mx
  const sh = height + 2 * my

  // MediaPipe の正規化座標はクロップキャンバス上の (0〜1)
  // cropFaceRaw は sw × sh → size × size にストレッチするので
  // 逆変換: pixel_x = sx + nx * sw, pixel_y = sy + ny * sh
  const toPixel = (nx: number, ny: number) => ({
    x: sx + nx * sw,
    y: sy + ny * sh,
  })

  return {
    rightEye: toPixel(lm.rightEye.x, lm.rightEye.y),
    leftEye:  toPixel(lm.leftEye.x,  lm.leftEye.y),
    noseTip:  toPixel(lm.noseTip.x,  lm.noseTip.y),
    mouth:    toPixel(lm.mouth.x,    lm.mouth.y),
    rightEar: toPixel(lm.rightEar.x, lm.rightEar.y),
    leftEar:  toPixel(lm.leftEar.x,  lm.leftEar.y),
  }
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const exps = new Float32Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - max); sum += exps[i] }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum
  return exps
}

function argmax(arr: Float32Array): number {
  let idx = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[idx]) idx = i
  return idx
}

/**
 * 静止画用: ImageBitmap + 検出結果を受け取り、1回だけ推論して RecognitionResult[] を返す。
 *
 * @param landmarksList 各検出に対応する5点ランドマーク（null = 取得失敗 or 小さすぎてスキップ）
 */
export async function runStaticImageClassification(
  config: ModelConfig,
  bitmap: ImageBitmap,
  detections: FaceDetection[],
  confidenceThreshold = 0.5,
  landmarksList: (FaceLandmarks | null)[] = [],
  minFacePx = 48,
): Promise<RecognitionResult[]> {
  if (detections.length === 0) return []

  const { session, classNames, mean, std, inputSize, hasAnomalyOutput, anomalyThreshold } = config
  const numClasses = classNames.length

  const tooSmallFlags = detections.map(d => isFaceTooSmall(d, minFacePx))

  // 推論対象のインデックス（小さすぎる顔を除外）
  const validIndices = detections.map((_, i) => i).filter(i => !tooSmallFlags[i])

  // 全て小さすぎる場合はそのままスキップ結果を返す
  if (validIndices.length === 0) {
    return detections.map((detection, i) => ({
      detection,
      classification: {
        predClassIdx: -1,
        identity: null,
        confidence: 0,
        anomalyScore: null,
        isAnomaly: false,
        unknownReason: 'too_small' as UnknownReason,
        tooSmall: true,
      } as ClassificationResult,
      landmarks: landmarksList[i]
        ? landmarksToPixel(landmarksList[i]!, detection)
        : null,
    }))
  }

  // 有効な顔のみクロップ・テンソル化
  const allTensors = validIndices.map(i => {
    const lm = landmarksList[i] ?? null
    const cropped = cropFace(bitmap, detections[i], inputSize, lm)
    return canvasToTensor(cropped, mean, std)
  })

  const B = validIndices.length
  const batchData = stackTensors(allTensors, inputSize, inputSize)
  const inputTensor = new ort.Tensor('float32', batchData, [B, 3, inputSize, inputSize])
  const inputName = session.inputNames[0]
  const outputs = await session.run({ [inputName]: inputTensor })

  const logitsAll = outputs[session.outputNames[0]].data as Float32Array
  const anomalyAll = hasAnomalyOutput
    ? (outputs['anomaly_score'].data as Float32Array)
    : null

  // 推論結果をインデックスに対応付け
  const validResults = new Map<number, ClassificationResult>()
  validIndices.forEach((origIdx, batchIdx) => {
    const logits = logitsAll.slice(batchIdx * numClasses, (batchIdx + 1) * numClasses)
    const probs = softmax(logits)
    const predClassIdx = argmax(probs)
    const confidence = probs[predClassIdx]
    const anomalyScore = anomalyAll ? anomalyAll[batchIdx] : null
    const isAnomaly =
      anomalyScore !== null && anomalyThreshold !== null
        ? anomalyScore > anomalyThreshold
        : false
    const identity =
      !isAnomaly && confidence >= confidenceThreshold
        ? classNames[predClassIdx]
        : null

    let unknownReason: UnknownReason | undefined
    if (identity === null) {
      if (isAnomaly) {
        unknownReason = 'anomaly'
      } else if (!hasAnomalyOutput) {
        unknownReason = 'no_anomaly_model'
      } else {
        unknownReason = 'low_confidence'
      }
    }

    validResults.set(origIdx, {
      predClassIdx,
      identity,
      confidence,
      anomalyScore,
      isAnomaly,
      unknownReason,
    })
  })

  // 全検出に対して結果を組み立て（小さすぎる顔は tooSmall フラグ付きで返す）
  return detections.map((detection, i) => {
    const pixelLandmarks = landmarksList[i]
      ? landmarksToPixel(landmarksList[i]!, detection)
      : null

    if (tooSmallFlags[i]) {
      return {
        detection,
        classification: {
          predClassIdx: -1,
          identity: null,
          confidence: 0,
          anomalyScore: null,
          isAnomaly: false,
          unknownReason: 'too_small' as UnknownReason,
          tooSmall: true,
        } as ClassificationResult,
        landmarks: pixelLandmarks,
      }
    }
    return { detection, classification: validResults.get(i)!, landmarks: pixelLandmarks }
  })
}
