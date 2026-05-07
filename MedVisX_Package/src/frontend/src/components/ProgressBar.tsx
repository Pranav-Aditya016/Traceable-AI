import { Check, ScanLine, Image, Brain } from 'lucide-react'
import type { PipelineStep } from '../types'

interface Props {
  step: PipelineStep
}

const STEPS = [
  { key: 'ocr', label: 'Text Extraction', icon: ScanLine },
  { key: 'generation', label: 'Image Generation', icon: Image },
  { key: 'xai', label: 'Explainability', icon: Brain },
]

export default function ProgressBar({ step }: Props) {
  const stepOrder = ['ocr', 'generation', 'xai', 'done']
  const currentIdx = stepOrder.indexOf(step)

  return (
    <div className="progress-section">
      <div className="progress-steps">
        {STEPS.map((s, i) => {
          const isDone = currentIdx > i
          const isActive = stepOrder[i] === step
          const Icon = s.icon

          return (
            <div
              key={s.key}
              className={`progress-step ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}
            >
              <div className="step-circle">
                {isDone ? <Check size={14} /> : <Icon size={14} />}
              </div>
              <span className="step-label">{s.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
