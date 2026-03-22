/// <reference types="vite/client" />

/** ビルド時に vite.config.ts の define で埋め込まれるモデルファイルサイズ (bytes) */
declare const __MODEL_FILE_SIZES__: Record<string, number>

/** ビルド時に vite.config.ts の define で埋め込まれる public/sample-images/ のファイル名一覧 */
declare const __SAMPLE_IMAGES__: string[]

/** Google Analytics gtag グローバル関数 */
declare function gtag(...args: unknown[]): void
