import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Hero } from './components/Hero';
import { UploadZone } from './components/UploadZone';
import { PipelineTracker } from './components/PipelineTracker';
import { SummaryView } from './components/SummaryView';
import { analyzeStream, type StageEvent } from './api/analyzeStream';
import type { PipelineStep, ReportData } from './types';

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 1, label: 'OCR Engine',                 model: 'TrOCR / PaliGemma 2 (local)',       status: 'pending', input: 'Uploaded image',         output: 'Raw OCR text' },
  { id: 2, label: 'Multimodal Analysis Brain',  model: 'MedGemma 4B (local)',                status: 'pending', input: 'OCR text + image',       output: 'Structured clinical data' },
  { id: 3, label: 'Prediction Scoring',         model: 'Symbolic scorer + SHAP (local)',     status: 'pending', input: 'Conditions + symptoms',  output: 'Ranked hypothesis' },
  { id: 4, label: 'Image Generation',           model: 'Stable Diffusion v1.5 (local)',      status: 'pending', input: 'Organ + condition',      output: 'Synthetic medical image' },
  { id: 5, label: 'Explainable AI Engine',      model: 'DenseNet121 Grad-CAM (local)',       status: 'pending', input: 'Synthetic image',        output: 'Activation heatmap' },
  { id: 6, label: 'Natural Language Explanation', model: 'MedGemma 4B + template fallback', status: 'pending', input: 'Full pipeline JSON',     output: 'Clinical narrative + audit metadata' },
];

const SunIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
  </svg>
);
const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

function App() {
  const [steps, setSteps]        = useState<PipelineStep[]>(PIPELINE_STEPS);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectedBadge, setDetectedBadge] = useState('');
  const [report, setReport]      = useState<ReportData | null>(null);
  const [error, setError]        = useState<string | null>(null);
  const [progress, setProgress]  = useState(0);
  const pipelineRef              = useRef<HTMLDivElement>(null);
  const cancelRef                = useRef<(() => void) | null>(null);

  // Theme
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('tracemd-theme');
    return (saved === 'dark' || saved === 'light') ? saved : 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tracemd-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  const handleReset = useCallback(() => {
    if (cancelRef.current) cancelRef.current();
    setSteps(PIPELINE_STEPS.map((s) => ({ ...s, status: 'pending', result: undefined })));
    setIsAnalyzing(false);
    setDetectedBadge('');
    setReport(null);
    setError(null);
    setProgress(0);
  }, []);

  const handleUpload = useCallback((file: File) => {
    handleReset();
    setIsAnalyzing(true);
    setError(null);
    setProgress(2);

    setTimeout(() => {
      pipelineRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

    const handleEvent = (event: StageEvent) => {
      const { stage, status, result, label, payload } = event;

      // Error from pipeline
      if (stage === -1 && status === 'error') {
        setSteps((prev) =>
          prev.map((s) => (s.status === 'running' ? { ...s, status: 'error', result } : s))
        );
        setError(result || 'Pipeline failed');
        setIsAnalyzing(false);
        setProgress(0);
        return;
      }

      // Update step status
      setSteps((prev) =>
        prev.map((s) => {
          if (s.id === stage) {
            return { ...s, status: status as PipelineStep['status'], result: result || label || s.result };
          }
          return s;
        })
      );

      // Progress bar
      if (status === 'running') {
        setProgress(Math.round(((stage - 1) / 6) * 100) + 2);
      } else if (status === 'complete') {
        setProgress(Math.round((stage / 6) * 100));
      }

      // Detect input type badge
      if (stage === 1 && status === 'complete' && payload && 'input_type' in payload) {
        setDetectedBadge('OCR — Gemma 4 (Ollama)');
      }
    };

    const handleComplete = (reportData: ReportData) => {
      setReport(reportData);
      setIsAnalyzing(false);
      setProgress(100);
    };

    const handleError = (message: string) => {
      setError(message);
      setSteps((prev) =>
        prev.map((s) => (s.status === 'running' ? { ...s, status: 'error', result: message } : s))
      );
      setIsAnalyzing(false);
      setProgress(0);
    };

    cancelRef.current = analyzeStream(file, handleEvent, handleComplete, handleError);
  }, [handleReset]);

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] transition-colors duration-300">
      <button
        type="button"
        onClick={toggleTheme}
        className="theme-toggle fixed top-4 right-4 z-50"
        title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      >
        {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
      </button>

      <AnimatePresence mode="wait">
        {report ? (
          <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <main className="max-w-7xl mx-auto px-4 py-6 md:px-8 md:py-8">
              <SummaryView data={report} onReset={handleReset} />
            </main>
            <footer className="border-t border-[var(--border-color)] py-6 px-4 text-center text-[var(--text-secondary)] text-sm">
              <p>TraceMD · Research Prototype · Not for clinical diagnostic use</p>
              <p className="mt-1 text-xs opacity-60">IEEE VLSI SATA 2026 · AIMLA 2026 · Paper ID 1314</p>
            </footer>
          </motion.div>
        ) : (
          <motion.div key="landing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Hero />

            <section id="upload" className="relative z-10 max-w-3xl mx-auto px-4 -mt-16 mb-20">
              <UploadZone onUpload={handleUpload} detectedBadge={detectedBadge} />
            </section>

            <div ref={pipelineRef}>
              {isAnalyzing && (
                <section className="max-w-3xl mx-auto px-4 mb-20">
                  <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-[var(--text-primary)]">Pipeline Progress</span>
                      <span className="text-sm font-bold text-[var(--accent)]">{progress}%</span>
                    </div>
                    <div className="w-full h-3 bg-[var(--border-color)] rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ease-out ${
                          progress < 100 ? 'bg-[var(--accent)] progress-bar-animated' : 'bg-[var(--success)]'
                        }`}
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-[var(--text-secondary)] mt-1">
                      {progress < 100 ? 'Processing your document through the AI pipeline…' : 'Pipeline complete!'}
                    </p>
                  </div>
                  <PipelineTracker steps={steps} />
                </section>
              )}

              {error && !isAnalyzing && (
                <section className="max-w-3xl mx-auto px-4 mb-20">
                  <div className="p-5 bg-[var(--danger)]/10 border border-[var(--danger)]/30 rounded-xl">
                    <h3 className="text-lg font-bold text-[var(--danger)] mb-2">Pipeline Error</h3>
                    <pre className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">{error}</pre>
                    <button type="button" onClick={handleReset} className="mt-4 px-6 py-2 bg-[var(--accent)] text-white font-semibold rounded-lg hover:bg-[var(--accent-light)] transition">
                      Try Again
                    </button>
                  </div>
                </section>
              )}
            </div>

            {!isAnalyzing && !error && (
              <section className="max-w-4xl mx-auto px-4 mb-20">
                <h2 className="text-3xl font-bold text-center mb-12">How it works</h2>
                <div className="space-y-6">
                  {PIPELINE_STEPS.map((step) => (
                    <motion.div key={step.id} initial={{ opacity: 0, x: step.id % 2 === 0 ? 50 : -50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, margin: '-40px' }} transition={{ duration: 0.5 }} className="p-6 bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl">
                      <div className="text-xs font-mono text-[var(--accent)] mb-2 uppercase tracking-wider">Stage {step.id}</div>
                      <h3 className="text-lg font-bold mb-2">{step.model}</h3>
                      <p className="text-sm text-[var(--text-secondary)]"><strong>Input:</strong> {step.input}</p>
                      <p className="text-sm text-[var(--text-secondary)]"><strong>Output:</strong> {step.output}</p>
                    </motion.div>
                  ))}
                </div>
              </section>
            )}

            <footer className="border-t border-[var(--border-color)] py-6 px-4 text-center text-[var(--text-secondary)] text-sm">
              <p>TraceMD · Research Prototype · Not for clinical diagnostic use</p>
              <p className="mt-1 text-xs opacity-60">IEEE VLSI SATA 2026 · AIMLA 2026 · Paper ID 1314</p>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
