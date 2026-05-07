import type { AnalysisResult } from './types'

const API_BASE = '/api'

export async function analyzeImages(
  files: File[],
  clinicalText: string,
  patientName: string,
  patientAge: string,
  patientSex: string,
  onProgress?: (step: string) => void,
): Promise<AnalysisResult> {
  const formData = new FormData()

  for (const file of files) {
    formData.append('files', file)
  }

  formData.append('clinical_text', clinicalText)
  formData.append('patient_name', patientName || 'Anonymous')
  formData.append('patient_age', patientAge)
  formData.append('patient_sex', patientSex)

  onProgress?.('Analyzing...')

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Error ${res.status}`)
  }

  return res.json()
}

export async function checkHealth(): Promise<{
  status: string
  device: string
  gpu: { name: string; vram_total_gb: number; vram_used_gb: number } | null
}> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error('Backend offline')
  return res.json()
}
