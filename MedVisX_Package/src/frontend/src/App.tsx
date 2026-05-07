import { useState, useEffect, useCallback } from 'react'
import Header from './components/Header'
import UploadPanel from './components/UploadPanel'
import ProgressBar from './components/ProgressBar'
import ResultsDashboard from './components/ResultsDashboard'
import LoadingOverlay from './components/LoadingOverlay'
import { analyzeImages, checkHealth } from './api'
import type { AnalysisResult, PipelineStep } from './types'

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [clinicalText, setClinicalText] = useState('')
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [patientSex, setPatientSex] = useState('')

  const [step, setStep] = useState<PipelineStep>('idle')
  const [loadingMsg, setLoadingMsg] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState('')

  const [backendOnline, setBackendOnline] = useState(false)
  const [gpuName, setGpuName] = useState('')

  // Health check
  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const h = await checkHealth()
        if (!cancelled) {
          setBackendOnline(true)
          setGpuName(h.gpu?.name || '')
        }
      } catch {
        if (!cancelled) setBackendOnline(false)
      }
    }
    check()
    const iv = setInterval(check, 15000)
    return () => { cancelled = true; clearInterval(iv) }
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!files.length && !clinicalText.trim()) {
      setError('Please upload images or enter clinical text.')
      return
    }
    setError('')
    setResult(null)
    setStep('ocr')
    setLoadingMsg('Analyzing document with LLaVA + Mistral...')

    try {
      // Simulate step progression (actual work is single request)
      const stepTimer = setTimeout(() => {
        setStep('generation')
        setLoadingMsg('Generating medical visualization with SDXL...')
      }, 8000)
      const stepTimer2 = setTimeout(() => {
        setStep('xai')
        setLoadingMsg('Computing explanations & XAI...')
      }, 30000)

      const data = await analyzeImages(
        files, clinicalText, patientName, patientAge, patientSex,
        (msg) => setLoadingMsg(msg),
      )

      clearTimeout(stepTimer)
      clearTimeout(stepTimer2)

      setResult(data)
      setStep('done')
      setLoadingMsg('')
    } catch (err: unknown) {
      clearTimeout(undefined) // ensure cleanup
      let message = 'Analysis failed.'
      if (err instanceof Error) message = err.message
      setError(message)
      setStep('error')
      setLoadingMsg('')
    }
  }, [files, clinicalText, patientName, patientAge, patientSex])

  const handleReset = useCallback(() => {
    setFiles([])
    setClinicalText('')
    setPatientName('')
    setPatientAge('')
    setPatientSex('')
    setStep('idle')
    setResult(null)
    setError('')
    setLoadingMsg('')
  }, [])

  const isLoading = step === 'ocr' || step === 'generation' || step === 'xai'

  return (
    <div className="app-layout">
      <Header online={backendOnline} gpuName={gpuName} />

      <main className="main-content">
        {/* Upload & Input */}
        {step === 'idle' || step === 'error' ? (
          <UploadPanel
            files={files}
            onFilesChange={setFiles}
            clinicalText={clinicalText}
            onClinicalTextChange={setClinicalText}
            patientName={patientName}
            onPatientNameChange={setPatientName}
            patientAge={patientAge}
            onPatientAgeChange={setPatientAge}
            patientSex={patientSex}
            onPatientSexChange={setPatientSex}
            onAnalyze={handleAnalyze}
            disabled={isLoading}
            error={error}
          />
        ) : null}

        {/* Progress */}
        {isLoading && <ProgressBar step={step} />}

        {/* Results */}
        {step === 'done' && result && (
          <ResultsDashboard result={result} onReset={handleReset} />
        )}
      </main>

      {/* Loading Overlay */}
      {isLoading && <LoadingOverlay message={loadingMsg} step={step} />}
    </div>
  )
}
