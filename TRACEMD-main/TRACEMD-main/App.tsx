import React, { useState, useCallback, useEffect } from 'react';
import { Header } from './components/Header';
import { FileUpload } from './components/FileUpload';
import { PipelineStatus } from './components/PipelineStatus';
import { SummaryView } from './components/SummaryView';
import { analyzeAndGenerate } from './services/geminiService';
import type { PipelineStatus as Status, ReportData } from './types';

// Extend window for AI Studio API
declare global {
  interface Window {
    aistudio: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

const App: React.FC = () => {
  const [pipelineStatus, setPipelineStatus] = useState<Status>('idle');
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resetState = () => {
    setPipelineStatus('idle');
    setReportData(null);
    setError(null);
  };

  const handleFileUpload = useCallback(async (file: File) => {
    resetState();
    setPipelineStatus('ocr'); // Combined analysis step

    try {
      // The new service function handles the entire pipeline in one go for efficiency
      const result = await analyzeAndGenerate(file, (status) => {
        // Update status as the pipeline progresses
        if (status === 'predicting') setPipelineStatus('predicting');
        if (status === 'generating') setPipelineStatus('generating');
      });
      
      setReportData(result);
      setPipelineStatus('complete');

    } catch (err) {
      console.error('Pipeline Error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during the pipeline processing.');
      setPipelineStatus('error');
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 font-sans">
      <Header />
      <main className="container mx-auto p-4 md:p-8">
        <div className="max-w-7xl mx-auto bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 md:p-8 space-y-8">
          
          {pipelineStatus !== 'complete' && (
            <div className="text-center">
              <h2 className="text-2xl md:text-3xl font-bold text-gray-700 dark:text-gray-200">Medical Report Analysis Pipeline</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">Upload a patient document (image) to start the AI-powered analysis.</p>
            </div>
          )}
          
          {pipelineStatus === 'idle' || pipelineStatus === 'error' ? (
            <FileUpload onFileUpload={handleFileUpload} />
          ) : pipelineStatus !== 'complete' ? (
            <div className="space-y-6">
                <PipelineStatus status={pipelineStatus} />
                <button
                    onClick={resetState}
                    className="w-full bg-gray-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-gray-600 transition duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
                    >
                    Cancel Analysis
                </button>
            </div>
          ) : null}

          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md" role="alert">
              <p className="font-bold">Error</p>
              <p>{error}</p>
              <button
                onClick={resetState}
                className="mt-4 bg-red-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-red-700 transition duration-300"
              >
                Start Over
              </button>
            </div>
          )}

          {/* Integration point for the new SummaryView component */}
          {/* This view is rendered once the entire pipeline is complete, displaying all results and XAI artifacts. */}
          {pipelineStatus === 'complete' && reportData && (
            <SummaryView initialData={reportData} onReset={resetState} />
          )}
        </div>
      </main>

      <footer className="text-center p-4 text-gray-500 dark:text-gray-400 text-sm">
        <p>This is a research prototype. All generated content is synthetic and not for diagnostic use.</p>
      </footer>
    </div>
  );
};

export default App;
