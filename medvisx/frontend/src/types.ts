export interface Localization {
  mask: { x: number; y: number }[]
  bbox: number[]  // [x1, y1, x2, y2]
}

export interface Provenance {
  generator: string
  seed: number
  conditioning: {
    organ: string
    condition: string
  }
  createdAt: string
}

export interface AuditLogEntry {
  user: string
  action: string
  timestamp: string
}

export interface AnalysisResult {
  report_id: string
  status?: string
  patient: {
    name: string
    age: string
    sex: string
  }
  ocr: {
    final_text: string
    crnn_text: string
    easyocr_text: string
    method_used: string
  }
  entities: {
    symptoms: { text: string; score: number }[]
    medications: { text: string; score: number }[]
    conditions: { text: string; score: number }[]
  }
  prediction: {
    top_disease: string
    confidence: number
    modality: string
    organ?: string
    hypotheses: { disease: string; score: number; model?: string }[]
    localization?: Localization
  }
  images: {
    generated: string // base64
    heatmap: string   // base64
    shap_plot: string // base64
  }
  xai: {
    shap_values: { feature: string; value: string; impact: number }[]
    text_features?: { feature: string; weight: number }[]
    explanation: string
    naturalLanguageSummary?: string
  }
  provenance?: Provenance
  auditLog?: AuditLogEntry[]
  timings: {
    analysis: number
    generation: number
    xai: number
    total: number
  }
  created_at: string
}

export type PipelineStep = 'idle' | 'ocr' | 'generation' | 'xai' | 'done' | 'error'
