import * as ort from 'onnxruntime-web/all'
import { onnx } from 'onnx-proto'
import type { ModelConfig } from '../types'

export type ExecutionProvider = 'wasm' | 'webgpu'

// onnxruntime-web の WASM を CDN から取得
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/'
// proxy worker は使わない。
// proxy=true にすると WASM バックエンドの初期化が worker 側に委譲され、
// 同一ページで WebGPU セッションを作る際に "WebAssembly is not initialized yet"
// エラーが発生するため、常に false に固定する。
ort.env.wasm.proxy = false

/** ArrayBuffer から ONNX メタデータをパースして返す */
function parseOnnxMetadataFromBuffer(buf: Uint8Array): Record<string, string> {
  const model = onnx.ModelProto.decode(buf) as {
    metadataProps?: Array<{ key: string; value: string }>
  }
  const meta: Record<string, string> = {}
  for (const prop of model.metadataProps ?? []) {
    meta[prop.key] = prop.value
  }
  return meta
}

export async function loadModel(
  modelUrl: string,
  ep: ExecutionProvider = 'wasm',
): Promise<ModelConfig> {
  // モデルを 1 回だけ fetch して ArrayBuffer を両方に使いまわす
  const res = await fetch(modelUrl)
  if (!res.ok) {
    throw new Error(`Failed to fetch model: ${res.status} ${res.statusText} (${modelUrl})`)
  }
  const arrayBuffer = await res.arrayBuffer()

  // メタデータ解析（onnx-proto）と ORT セッション初期化を並行実行
  const buf = new Uint8Array(arrayBuffer)
  const [meta, session] = await Promise.all([
    Promise.resolve(parseOnnxMetadataFromBuffer(buf)),
    ort.InferenceSession.create(arrayBuffer, {
      executionProviders: [ep],
      graphOptimizationLevel: 'all',
    }),
  ])

  if (!meta['class_names']) {
    throw new Error("ONNX model does not contain 'class_names' in metadata.")
  }
  const classNames: string[] = JSON.parse(meta['class_names'])
  const mean: number[] = JSON.parse(meta['imagenet_mean'] ?? '[0.485,0.456,0.406]')
  const std: number[] = JSON.parse(meta['imagenet_std'] ?? '[0.229,0.224,0.225]')
  const inputSize: number = meta['input_size'] ? parseInt(meta['input_size']) : 224

  const hasAnomalyOutput = session.outputNames.includes('anomaly_score')

  // AnomalyClassifier 閾値: メタデータキー 'anomaly_threshold'
  // sigmoid 出力、anomaly_score > anomalyThreshold のとき異常
  const anomalyThreshold =
    hasAnomalyOutput && meta['anomaly_threshold']
      ? parseFloat(meta['anomaly_threshold'])
      : null

  console.info(
    `[modelLoader] classes=${classNames.length} inputSize=${inputSize} ep=${ep}`,
    `mean=${mean} std=${std}`,
    hasAnomalyOutput ? `anomaly_threshold=${anomalyThreshold}` : 'no anomaly'
  )

  return { session, classNames, mean, std, inputSize, hasAnomalyOutput, anomalyThreshold }
}
