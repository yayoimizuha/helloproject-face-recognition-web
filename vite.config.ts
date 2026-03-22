import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { statSync, readdirSync } from 'fs'
import { resolve } from 'path'

/** public/models/ 以下の .onnx ファイルサイズをビルド時に収集する */
function getModelFileSizes(): Record<string, number> {
  const files = [
    'model_best_with_anomaly.onnx',
    'model_best_with_anomaly_bf16.onnx',
    'model_best_with_anomaly_fp16.onnx',
    'model_best_with_anomaly_bf16int8.onnx',
    'model_best_with_anomaly_bf16fp8.onnx',
    'model_best_with_anomaly_fp16int8.onnx',
    'model_best_with_anomaly_fp16fp8.onnx',
    'yolov12n-face.onnx',
    'yolov12s-face.onnx',
    'yolov12m-face.onnx',
    'yolov12l-face.onnx',
  ]
  const sizes: Record<string, number> = {}
  for (const f of files) {
    try {
      sizes[f] = statSync(resolve(__dirname, 'public/models', f)).size
    } catch {
      sizes[f] = 0
    }
  }
  return sizes
}

/** public/sample-images/ 以下の画像ファイル名一覧をビルド時に収集する */
function getSampleImageFiles(): string[] {
  try {
    return readdirSync(resolve(__dirname, 'public/sample-images'))
      .filter(f => /\.(jpe?g|png|webp|gif)$/i.test(f))
      .sort()
  } catch {
    return []
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    // ビルド時にファイルサイズを定数として埋め込む
    __MODEL_FILE_SIZES__: JSON.stringify(getModelFileSizes()),
    // ビルド時にサンプル画像ファイル名一覧を埋め込む
    __SAMPLE_IMAGES__: JSON.stringify(getSampleImageFiles()),
  },
  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  // .onnx は public/ から静的配信するため assetsInclude に含めない
})
