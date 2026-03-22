import type { RecognitionResult } from '../types'

interface Props {
  results: RecognitionResult[]
}

export function RecognitionResultList({ results }: Props) {
  if (results.length === 0) return null
  return (
    <div className="flex flex-wrap gap-2 p-2">
      {results.map((r, i) => {
        const c = r.classification

        let badgeClass: string
        let label: string

        if (c.identity) {
          badgeClass = 'badge-success'
          label = `${c.identity} ${(c.confidence * 100).toFixed(1)}%`
        } else {
          badgeClass = c.unknownReason === 'anomaly' ? 'badge-error' : 'badge-warning'
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

        return (
          <span key={i} className={`badge ${badgeClass} text-xs`}>{label}</span>
        )
      })}
    </div>
  )
}
