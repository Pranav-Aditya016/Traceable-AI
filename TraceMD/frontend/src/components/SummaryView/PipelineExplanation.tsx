import React from 'react';
import type { ReportData } from '../../types';

interface PipelineExplanationProps {
  reportData: ReportData;
}

const Step: React.FC<{ icon: React.ReactElement; title: string; children: React.ReactNode }> = ({ icon, title, children }) => (
  <div className="flex items-start space-x-4">
    <div className="flex-shrink-0 w-11 h-11 flex items-center justify-center bg-[var(--accent)]/10 rounded-full text-[var(--accent-light)]">
      {icon}
    </div>
    <div>
      <h4 className="font-semibold text-[var(--text-primary)]">{title}</h4>
      <p className="text-sm text-[var(--text-secondary)] mt-1 leading-relaxed">{children}</p>
    </div>
  </div>
);

export const PipelineExplanation: React.FC<PipelineExplanationProps> = ({ reportData }) => {
  const { fields, predictions, xai } = reportData;
  const primary = predictions.conditions[0];

  return (
    <div className="p-5 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
      <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">How The AI Found The Result</h3>
      <div className="space-y-5">
        <Step icon={<IconDocument />} title="1. Document OCR">
          The uploaded document was processed using Optical Character Recognition (OCR) to extract the raw text.
        </Step>
        <Step icon={<IconExtract />} title="2. Key Field Extraction">
          The AI identified key patient information: Age ({fields.age}), Sex ({fields.sex}), and Symptoms ({fields.symptoms.join(', ')}).
        </Step>
        <Step icon={<IconPredict />} title="3. Condition Prediction">
          Based on the extracted data, the model '{primary?.model}' predicted a '{primary?.label}' in the '{predictions.organ}' with {primary ? (primary.confidence * 100).toFixed(0) : 'N/A'}% confidence.
        </Step>
        <Step icon={<IconLocalize />} title="4. Finding Localization">
          The system generated a segmentation mask and a bounding box to pinpoint the location of the predicted finding within the organ.
        </Step>
        <Step icon={<IconGenerate />} title="5. Synthetic Image Generation">
          A synthetic image was created by the '{xai.provenance.generator}' model, conditioned on the patient data and predicted findings to visualize the result.
        </Step>
        <Step icon={<IconXai />} title="6. XAI Analysis">
          Explainable AI methods were used to identify the top text features (e.g., '{xai.text_features[0]?.feature}') that influenced the prediction.
        </Step>
      </div>
    </div>
  );
};

// SVG Icons
const IconDocument = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
);
const IconExtract = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>
);
const IconPredict = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
);
const IconLocalize = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
);
const IconGenerate = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
);
const IconXai = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
);
