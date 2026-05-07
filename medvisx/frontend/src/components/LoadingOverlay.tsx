import type { PipelineStep } from '../types'

interface Props {
  message: string
  step: PipelineStep
}

export default function LoadingOverlay({ message, step }: Props) {
  const stepMessages: Record<string, string> = {
    ocr: 'Extracting text from medical documents...',
    generation: 'Generating synthetic medical visualization...',
    xai: 'Computing Grad-CAM heatmap & explanations...',
  }

  return (
    <div className="loading-overlay">
      <div className="loading-spinner-lg" />
      <div className="loading-text">
        {message || stepMessages[step] || 'Processing...'}
      </div>
      <div style={{
        fontSize: '0.75rem',
        color: 'var(--text-dim)',
        marginTop: 4,
      }}>
        This may take 1-3 minutes on first run while models load
      </div>
    </div>
  )
}
