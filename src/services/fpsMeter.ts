const WINDOW = 30  // 直近Nフレームで平均

export class FpsMeter {
  private times: number[] = []

  tick(): void {
    const now = performance.now()
    this.times.push(now)
    if (this.times.length > WINDOW) this.times.shift()
  }

  get fps(): number {
    if (this.times.length < 2) return 0
    const dt = this.times[this.times.length - 1] - this.times[0]
    return dt > 0 ? ((this.times.length - 1) / dt) * 1000 : 0
  }

  reset(): void { this.times = [] }
}
