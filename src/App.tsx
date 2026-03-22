import {
  useEffect, useRef, useState, useCallback,
} from 'react'
import {
  ImageIcon, Upload, Loader, Settings, Play, HelpCircle, Images,
} from 'lucide-react'

import type { RecognitionResult, FaceDetection, ModelConfig } from './types'
import { loadModel, type ExecutionProvider } from './services/modelLoader'
import { YoloFaceDetector, type YoloModelSize } from './services/detectors/YoloFaceDetector'
import { runStaticImageClassification } from './services/faceRecognition'
import { MediaPipeLandmarker } from './services/MediaPipeLandmarker'
import { cropFaceRaw } from './services/faceAlignment'
import { FaceOverlayCanvas } from './components/FaceOverlayCanvas'
import { RecognitionResultList } from './components/RecognitionResultList'

/** public/sample-images/ のファイル名から表示ラベルを生成（"名前=グループ=..." → "名前 (グループ)"） */
function sampleImageLabel(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, '') // 拡張子除去
  const parts = base.split('=')
  if (parts.length >= 2) return `${parts[0]} (${parts[1]})`
  return parts[0]
}

/** ビルド時に埋め込まれたサンプル画像一覧 */
const SAMPLE_IMAGES: { filename: string; url: string; label: string }[] =
  __SAMPLE_IMAGES__.map(f => ({
    filename: f,
    url: `/sample-images/${encodeURI(f)}`,
    label: sampleImageLabel(f),
  }))

/** バイト数を "X.XMB" 形式に変換 */
function fmtMB(filename: string): string {
  const bytes = __MODEL_FILE_SIZES__[filename] ?? 0
  return bytes > 0 ? `${(bytes / 1024 / 1024).toFixed(1)}MB` : '?MB'
}

// ── 認識モデル選択肢 ────────────────────────────────────────────────────────
type RecogModelKey =
  | 'fp32' | 'bf16' | 'fp16'
  | 'bf16int8' | 'bf16fp8' | 'fp16int8' | 'fp16fp8'

const RECOG_MODELS: { value: RecogModelKey; label: string; desc: string; url: string }[] = [
  { value: 'fp32',     label: `FP32 (${fmtMB('model_best_with_anomaly.onnx')})`,               desc: '最高精度・最大サイズ',           url: '/models/model_best_with_anomaly.onnx' },
  { value: 'bf16',     label: `BF16 (${fmtMB('model_best_with_anomaly_bf16.onnx')})`,          desc: 'BFloat16 全体変換',              url: '/models/model_best_with_anomaly_bf16.onnx' },
  { value: 'fp16',     label: `FP16 (${fmtMB('model_best_with_anomaly_fp16.onnx')})`,          desc: 'Float16 全体変換',               url: '/models/model_best_with_anomaly_fp16.onnx' },
  { value: 'bf16int8', label: `BF16+INT8 (${fmtMB('model_best_with_anomaly_bf16int8.onnx')})`, desc: 'BF16 + per-block INT8量子化',    url: '/models/model_best_with_anomaly_bf16int8.onnx' },
  { value: 'bf16fp8',  label: `BF16+FP8 (${fmtMB('model_best_with_anomaly_bf16fp8.onnx')})`,  desc: 'BF16 + per-block FP8量子化',     url: '/models/model_best_with_anomaly_bf16fp8.onnx' },
  { value: 'fp16int8', label: `FP16+INT8 (${fmtMB('model_best_with_anomaly_fp16int8.onnx')})`, desc: 'FP16 + per-block INT8量子化',    url: '/models/model_best_with_anomaly_fp16int8.onnx' },
  { value: 'fp16fp8',  label: `FP16+FP8 (${fmtMB('model_best_with_anomaly_fp16fp8.onnx')})`,  desc: 'FP16 + per-block FP8量子化',     url: '/models/model_best_with_anomaly_fp16fp8.onnx' },
]

// ── YOLO-Face モデル選択肢 ───────────────────────────────────────────────────
const YOLO_MODELS: { value: YoloModelSize; label: string; desc: string }[] = [
  { value: 'n', label: `Nano   (${fmtMB('yolov12n-face.onnx')})`, desc: '最速・軽量（デフォルト）' },
  { value: 's', label: `Small  (${fmtMB('yolov12s-face.onnx')})`, desc: 'バランス型' },
  { value: 'm', label: `Medium (${fmtMB('yolov12m-face.onnx')})`, desc: '高精度' },
  { value: 'l', label: `Large  (${fmtMB('yolov12l-face.onnx')})`, desc: '最高精度・重い' },
]

// ── 実行バックエンド ─────────────────────────────────────────────────────────
const EP_OPTIONS: { value: ExecutionProvider; label: string; desc: string }[] = [
  { value: 'wasm',   label: 'WebAssembly', desc: 'CPU上でSIMD実行。最も互換性が高い' },
  { value: 'webgpu', label: 'WebGPU',      desc: 'GPU上でWebGPU Computeを使って実行（最速）' },
]

type AppStatus = 'idle' | 'loading-model' | 'loading-detect' | 'ready' | 'error'

export default function App() {
  // ── モデル・検出器 ──────────────────────────────────────────────────────────
  const [model, setModel] = useState<ModelConfig | null>(null)
  const [appStatus, setAppStatus] = useState<AppStatus>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  // ── 設定 ────────────────────────────────────────────────────────────────────
  const [ep, setEp] = useState<ExecutionProvider>('wasm')
  const [recogModelKey, setRecogModelKey] = useState<RecogModelKey>('bf16fp8')
  const [yoloSize, setYoloSize] = useState<YoloModelSize>('n')
  const [detectThreshold, setDetectThreshold] = useState(0.40)   // YOLO 検出閾値
  const [recogThreshold, setRecogThreshold] = useState(0.60)     // 顔認識 confidence 閾値
  const [minFacePx, setMinFacePx] = useState(48)                 // Too Small 判定ピクセル数
  const [showLandmarks, setShowLandmarks] = useState(false)       // ランドマーク表示
  const [samplesModalOpen, setSamplesModalOpen] = useState(false) // サンプル画像モーダル

  // ── 画像 ────────────────────────────────────────────────────────────────────
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageReady, setImageReady] = useState(false)
  const [displaySize, setDisplaySize] = useState({ w: 640, h: 480 })
  const [sourceSize, setSourceSize] = useState({ w: 640, h: 480 })
  const imgRef = useRef<HTMLImageElement>(null)

  // ── 推論結果 ────────────────────────────────────────────────────────────────
  const [results, setResults] = useState<RecognitionResult[]>([])
  const [detections, setDetections] = useState<FaceDetection[]>([])
  const [faceCount, setFaceCount] = useState(0)
  const [detectMs, setDetectMs] = useState<number | null>(null)
  const [recognizeMs, setRecognizeMs] = useState<number | null>(null)

  // ── refs ────────────────────────────────────────────────────────────────────
  const detectorRef = useRef<YoloFaceDetector | null>(null)
  const landmarkerRef = useRef<MediaPipeLandmarker | null>(null)
  const modelRef = useRef<ModelConfig | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const epRef = useRef(ep)
  const yoloSizeRef = useRef(yoloSize)
  const recogModelKeyRef = useRef(recogModelKey)

  useEffect(() => { epRef.current = ep }, [ep])
  useEffect(() => { yoloSizeRef.current = yoloSize }, [yoloSize])
  useEffect(() => { recogModelKeyRef.current = recogModelKey }, [recogModelKey])

  // 初回訪問時に説明モーダルを自動表示
  useEffect(() => {
    const STORAGE_KEY = 'face_recognition_visited'
    if (!localStorage.getItem(STORAGE_KEY)) {
      localStorage.setItem(STORAGE_KEY, '1')
      const dialog = document.getElementById('modal-about') as HTMLDialogElement | null
      dialog?.showModal()
    }
  }, [])

  // アンマウント時クリーンアップ
  useEffect(() => {
    return () => {
      detectorRef.current?.dispose()
      landmarkerRef.current?.dispose()
      modelRef.current?.session.release().catch(() => {})
    }
  }, [])

  // ── 設定変更時：ロード済みモデル・検出器を破棄 ──────────────────────────
  const invalidateModels = useCallback(() => {
    if (modelRef.current) {
      modelRef.current.session.release().catch(() => {})
      modelRef.current = null
      setModel(null)
    }
    if (detectorRef.current) {
      detectorRef.current.dispose()
      detectorRef.current = null
    }
    setResults([])
    setDetections([])
    setFaceCount(0)
    setDetectMs(null)
    setRecognizeMs(null)
    setAppStatus('idle')
  }, [])

  const handleRecogModelChange = useCallback((key: RecogModelKey) => {
    setRecogModelKey(key)
    invalidateModels()
  }, [invalidateModels])

  const handleYoloSizeChange = useCallback((size: YoloModelSize) => {
    setYoloSize(size)
    if (detectorRef.current) {
      detectorRef.current.dispose()
      detectorRef.current = null
    }
    setResults([])
    setDetections([])
    setFaceCount(0)
    setDetectMs(null)
    setRecognizeMs(null)
    setAppStatus(model ? 'ready' : 'idle')
  }, [model])

  const handleEpChange = useCallback((newEp: ExecutionProvider) => {
    setEp(newEp)
    invalidateModels()
  }, [invalidateModels])

  // ── 画像ファイル選択 ────────────────────────────────────────────────────────
  const handleImageFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file))
    setImageReady(false)
    setResults([])
    setDetections([])
    setFaceCount(0)
    setDetectMs(null)
    setRecognizeMs(null)
    setAppStatus('idle')
    e.target.value = ''
  }, [imageUrl])

  // ── サンプル画像選択 ────────────────────────────────────────────────────────
  const handleSampleImage = useCallback((url: string) => {
    if (imageUrl?.startsWith('blob:')) URL.revokeObjectURL(imageUrl)
    setImageUrl(url)
    setImageReady(false)
    setResults([])
    setDetections([])
    setFaceCount(0)
    setDetectMs(null)
    setRecognizeMs(null)
    setAppStatus('idle')
  }, [imageUrl])

  // 画像ロード完了 → 表示サイズだけ計算
  const handleImageLoad = useCallback(() => {
    const img = imgRef.current
    if (!img) return
    const sw = img.naturalWidth
    const sh = img.naturalHeight
    setSourceSize({ w: sw, h: sh })
    const MAX_DISPLAY = 1200
    const ratio = Math.min(MAX_DISPLAY / sw, MAX_DISPLAY / sh, 1)
    setDisplaySize({ w: Math.round(sw * ratio), h: Math.round(sh * ratio) })
    setImageReady(true)
  }, [])

  // ── 「実行」ボタン ────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    const img = imgRef.current
    if (!img) return

    setResults([])
    setDetections([])
    setFaceCount(0)
    setDetectMs(null)
    setRecognizeMs(null)

    const imgWidth = img.naturalWidth
    const imgHeight = img.naturalHeight

    try {
      if (!modelRef.current) {
        setAppStatus('loading-model')
        const url = RECOG_MODELS.find(m => m.value === recogModelKeyRef.current)!.url
        const m = await loadModel(url, epRef.current)
        modelRef.current = m
        setModel(m)
      }

      setAppStatus('loading-detect')

      if (!detectorRef.current) {
        const det = new YoloFaceDetector({ modelSize: yoloSizeRef.current, ep: epRef.current })
        await det.initialize()
        detectorRef.current = det
      }

      const m = modelRef.current!
      const det = detectorRef.current!
      const bitmap = await createImageBitmap(img)

      const t0detect = performance.now()
      const dets = await det.detect(bitmap, detectThreshold)
      const detectElapsed = performance.now() - t0detect
      setDetectMs(detectElapsed)
      setDetections(dets)
      setFaceCount(dets.length)

      if (!landmarkerRef.current) {
        const lm = new MediaPipeLandmarker()
        await lm.initialize()
        landmarkerRef.current = lm
      }
      const lm = landmarkerRef.current
      const LANDMARK_CROP_SIZE = 192
      const landmarksList = dets.map(d => {
        const crop = cropFaceRaw(bitmap, d, LANDMARK_CROP_SIZE)
        return lm.detectLandmarks(crop)
      })

      const t0recog = performance.now()
      const recResults = await runStaticImageClassification(
        m, bitmap, dets, recogThreshold, landmarksList, minFacePx,
      )
      const recognizeElapsed = performance.now() - t0recog
      setRecognizeMs(recognizeElapsed)
      bitmap.close()

      setResults(recResults)
      setAppStatus('ready')

      // GA4 イベント計測
      gtag('event', 'run_inference', {
        recog_model: recogModelKeyRef.current,
        yolo_size: yoloSizeRef.current,
        backend: epRef.current,
        image_width: imgWidth,
        image_height: imgHeight,
        face_count: dets.length,
        detect_ms: Math.round(detectElapsed),
        recognize_ms: Math.round(recognizeElapsed),
        total_ms: Math.round(detectElapsed + recognizeElapsed),
      })
    } catch (err) {
      setErrorMsg(String(err))
      setAppStatus('error')
      gtag('event', 'run_inference_error', {
        recog_model: recogModelKeyRef.current,
        yolo_size: yoloSizeRef.current,
        backend: epRef.current,
        error_message: String(err),
      })
    }
  }, [detectThreshold, recogThreshold, minFacePx])

  const isLoading = appStatus === 'loading-model' || appStatus === 'loading-detect'
  const canRun = imageReady && !isLoading

  return (
    <div className="min-h-screen bg-base-300 flex flex-col items-center gap-3 pb-8">

      {/* ── ヘッダー ───────────────────────────────────────────────────── */}
      <header className="w-full bg-base-100 shadow">
        <div className="max-w-3xl mx-auto px-4 py-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold">顔認識AI</span>
              {/* 説明ポップアップ */}
              <button
                className="btn btn-ghost btn-xs btn-circle opacity-50 hover:opacity-100"
                onClick={() => (document.getElementById('modal-about') as HTMLDialogElement)?.showModal()}
                aria-label="このサイトについて"
              >
                <HelpCircle size={16} />
              </button>
            </div>
            <div className="flex items-center gap-1.5 flex-wrap justify-end">
              {faceCount > 0 && (
                <span className="badge badge-info badge-sm font-mono">{faceCount} 顔</span>
              )}
              {detectMs !== null && (
                <span className="badge badge-ghost badge-sm font-mono" title="検出時間">
                  検出 {detectMs.toFixed(0)}ms
                </span>
              )}
              {recognizeMs !== null && (
                <span className="badge badge-ghost badge-sm font-mono" title="認識時間">
                  認識 {recognizeMs.toFixed(0)}ms
                </span>
              )}
            </div>
          </div>
          {model && (
            <p className="text-xs opacity-40 mt-0.5 truncate">
              {model.classNames.length} クラス · {model.inputSize}px
            </p>
          )}
        </div>
      </header>

      {/* ── 説明モーダル ─────────────────────────────────────────────────── */}
      <dialog id="modal-about" className="modal modal-bottom sm:modal-middle">
        <div className="modal-box">
          <h3 className="font-bold text-lg mb-3">顔認識AIについて</h3>
          <div className="flex flex-col gap-2 text-sm opacity-80">
            <p>
              ブラウザ上で完全にローカル動作する顔認識デモです。
              アップロードした画像はサーバーに送信されず、すべての推論はデバイス上で完結します。
            </p>
            <p>
              <strong>顔検出</strong>に YOLOv12-Face、<strong>顔認識</strong>に独自学習した分類モデル、
              <strong>顔アライメント</strong>に MediaPipe BlazeFace を使用しています。
              推論エンジンには <strong>ONNX Runtime Web</strong> を採用しており、WebAssembly または WebGPU バックエンドで動作します。
            </p>
            <p>
              異常検知スコア（Anomaly Score）が閾値を超えた顔は
              登録外の人物として <span className="text-error font-semibold">Anomaly</span> 判定されます。
            </p>
            <p className="text-warning">
              ブラウザ上での WASM 実行はネイティブ環境と比べて推論速度が大幅に低下します。
              特にモデルサイズが大きい場合や WebGPU が利用できない環境では、処理に数秒〜十数秒かかることがあります。
            </p>
            <div className="mt-1 flex flex-col gap-1">
              <div className="flex gap-2 items-center">
                <span className="badge badge-success badge-sm">名前 XX%</span>
                <span>登録済みの人物として認識</span>
              </div>
              <div className="flex gap-2 items-center">
                <span className="badge badge-error badge-sm">Anomaly</span>
                <span>登録外の人物（異常スコア超過）</span>
              </div>
              <div className="flex gap-2 items-center">
                <span className="badge badge-warning badge-sm">Low Conf</span>
                <span>信頼度が低く認識できなかった</span>
              </div>
              <div className="flex gap-2 items-center">
                <span className="badge badge-warning badge-sm">Too Small</span>
                <span>顔が小さすぎて推論をスキップ</span>
              </div>
            </div>
          </div>
          <div className="modal-action">
            <form method="dialog">
              <button className="btn btn-sm">閉じる</button>
            </form>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button>close</button>
        </form>
      </dialog>

      {/* ── サンプル画像モーダル ──────────────────────────────────────────── */}
      <dialog className={`modal modal-bottom sm:modal-middle${samplesModalOpen ? ' modal-open' : ''}`}>
        <div className="modal-box w-full max-w-2xl">
          <h3 className="font-bold text-lg mb-3">サンプル画像</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {/* モーダルが開いているときだけ img をレンダリングしてロードを遅延 */}
            {samplesModalOpen && SAMPLE_IMAGES.map(s => (
              <button
                key={s.filename}
                className="flex flex-col gap-1 rounded-lg overflow-hidden border border-base-300 hover:border-primary transition-colors cursor-pointer bg-base-200"
                onClick={() => {
                  handleSampleImage(s.url)
                  setSamplesModalOpen(false)
                }}
              >
                <img
                  src={s.url}
                  alt={s.label}
                  className="w-full aspect-square object-cover"
                />
                <span className="text-xs px-2 py-1 text-center truncate w-full opacity-70">{s.label}</span>
              </button>
            ))}
          </div>
          <div className="modal-action">
            <button className="btn btn-sm" onClick={() => setSamplesModalOpen(false)}>閉じる</button>
          </div>
        </div>
        <div className="modal-backdrop" onClick={() => setSamplesModalOpen(false)} />
      </dialog>

      {/* ── メインコンテンツ（PC: max-w-3xl 中央寄せ、スマホ: 全幅） ── */}
      <div className="w-full max-w-3xl mx-auto flex flex-col gap-3 px-0 sm:px-3">

        {/* ── ビューポート ─────────────────────────────────────────────── */}
        <div
          className="relative w-full bg-base-200 sm:rounded-xl overflow-hidden flex items-center justify-center"
          style={{ aspectRatio: imageReady ? `${displaySize.w}/${displaySize.h}` : '4/3', minHeight: 200 }}
        >
          {imageUrl ? (
            <img
              ref={imgRef}
              src={imageUrl}
              alt="input"
              onLoad={handleImageLoad}
              className="w-full h-full object-contain"
            />
          ) : (
            <label
              className="text-base-content/30 flex flex-col items-center gap-3 py-16 cursor-pointer hover:text-base-content/50 transition-colors w-full h-full justify-center"
              onClick={() => !isLoading && fileInputRef.current?.click()}
            >
              <ImageIcon size={56} />
              <span className="text-sm">クリックして画像を選択</span>
            </label>
          )}

          {(results.length > 0 || detections.length > 0) && (
            <FaceOverlayCanvas
              results={results}
              detections={detections}
              width={displaySize.w}
              height={displaySize.h}
              sourceWidth={sourceSize.w}
              sourceHeight={sourceSize.h}
              showLandmarks={showLandmarks}
            />
          )}

          {isLoading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center flex-col gap-3">
              <Loader className="animate-spin text-white" size={40} />
              <span className="text-white text-sm">
                {appStatus === 'loading-model' ? 'モデルを読み込み中...' : '検出・認識中...'}
              </span>
            </div>
          )}
        </div>

        {/* ── 認識結果 ─────────────────────────────────────────────────── */}
        {results.length > 0 && (
          <div className="px-3 sm:px-0">
            <RecognitionResultList results={results} />
          </div>
        )}
        {imageReady && appStatus === 'ready' && detections.length === 0 && (
          <div className="badge badge-warning py-3 px-4 mx-auto">顔が検出されませんでした</div>
        )}

        {/* ── 設定パネル ───────────────────────────────────────────────── */}
        <div className="px-3 sm:px-0">
          <div className="card bg-base-100 shadow">
            <div className="card-body p-4 gap-4">

              <div className="flex items-center gap-2">
                <Settings size={14} />
                <span className="font-semibold text-sm">設定</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">認識モデル</span>
                  <select
                    className="select select-sm select-bordered w-full"
                    value={recogModelKey}
                    onChange={e => handleRecogModelChange(e.target.value as RecogModelKey)}
                    disabled={isLoading}
                  >
                    {RECOG_MODELS.map(o => (
                      <option key={o.value} value={o.value} title={o.desc}>{o.label}</option>
                    ))}
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">顔検出モデル (YOLO-Face)</span>
                  <select
                    className="select select-sm select-bordered w-full"
                    value={yoloSize}
                    onChange={e => handleYoloSizeChange(e.target.value as YoloModelSize)}
                    disabled={isLoading}
                  >
                    {YOLO_MODELS.map(o => (
                      <option key={o.value} value={o.value} title={o.desc}>{o.label}</option>
                    ))}
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">実行バックエンド</span>
                  <select
                    className="select select-sm select-bordered w-full"
                    value={ep}
                    onChange={e => handleEpChange(e.target.value as ExecutionProvider)}
                    disabled={isLoading}
                  >
                    {EP_OPTIONS.map(o => (
                      <option key={o.value} value={o.value} title={o.desc}>{o.label}</option>
                    ))}
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">検出閾値: {detectThreshold.toFixed(2)}</span>
                  <input
                    type="range" min="0.1" max="0.9" step="0.05"
                    value={detectThreshold}
                    onChange={e => setDetectThreshold(Number(e.target.value))}
                    className="range range-sm range-primary mt-1"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">認識閾値: {recogThreshold.toFixed(2)}</span>
                  <input
                    type="range" min="0.1" max="0.99" step="0.01"
                    value={recogThreshold}
                    onChange={e => setRecogThreshold(Number(e.target.value))}
                    className="range range-sm range-secondary mt-1"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-xs opacity-60">Too Small 閾値: {minFacePx}px</span>
                  <input
                    type="range" min="16" max="128" step="4"
                    value={minFacePx}
                    onChange={e => setMinFacePx(Number(e.target.value))}
                    className="range range-sm range-accent mt-1"
                  />
                </div>

                <div className="flex items-center gap-2 self-end pb-1">
                  <input
                    id="toggle-landmarks"
                    type="checkbox"
                    className="toggle toggle-sm toggle-primary"
                    checked={showLandmarks}
                    onChange={e => setShowLandmarks(e.target.checked)}
                  />
                  <label htmlFor="toggle-landmarks" className="text-xs opacity-60 cursor-pointer select-none">
                    ランドマーク表示
                  </label>
                </div>

              </div>

              <div className="flex gap-3">
                {/* サンプル画像ボタン */}
                {SAMPLE_IMAGES.length > 0 && (
                  <button
                    className="btn btn-outline btn-sm gap-2"
                    onClick={() => setSamplesModalOpen(true)}
                    disabled={isLoading}
                  >
                    <Images size={15} /> サンプル画像を利用
                  </button>
                )}
                <label className={`btn btn-outline btn-sm gap-2 cursor-pointer ${isLoading ? 'btn-disabled' : ''}`}>
                  <Upload size={15} /> 画像を選択
                  <input
                    ref={fileInputRef}
                    type="file" accept="image/*" className="hidden"
                    onChange={handleImageFile} disabled={isLoading}
                  />
                </label>
                <button
                  className="btn btn-primary btn-sm gap-2"
                  onClick={handleRun} disabled={!canRun}
                >
                  <Play size={15} /> 実行
                </button>
              </div>

            </div>
          </div>
        </div>

        {/* ── エラー ───────────────────────────────────────────────────── */}
        {appStatus === 'error' && (
          <div className="alert alert-error mx-3 sm:mx-0">
            <span className="text-sm">{errorMsg}</span>
            <button className="btn btn-sm btn-ghost ml-auto" onClick={() => setAppStatus('idle')}>
              閉じる
            </button>
          </div>
        )}

      </div>
    </div>
  )
}
