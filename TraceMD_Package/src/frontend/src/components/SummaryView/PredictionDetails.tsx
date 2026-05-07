import React from 'react';
import type { ReportData } from '../../types';

interface PredictionDetailsProps {
  predictions: ReportData['predictions'];
}

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => {
  const pct = Math.round(value * 100);
  let color = 'bg-[var(--success)]';
  if (pct < 75) color = 'bg-[var(--warning)]';
  if (pct < 50) color = 'bg-[var(--danger)]';

  return (
    <div className="w-full bg-[var(--bg-primary)] rounded-full h-2">
      <div
        className={`${color} h-2 rounded-full transition-all duration-500`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
};

export const PredictionDetails: React.FC<PredictionDetailsProps> = ({ predictions }) => {
  return (
    <div className="bg-[var(--bg-card)] p-4 rounded-lg border border-[var(--border-color)]">
      <h4 className="font-semibold text-[var(--text-primary)] mb-3">AI Prediction</h4>
      <div className="space-y-4">
        <div className="flex justify-between text-sm">
          <span className="text-[var(--text-secondary)]">Predicted Organ:</span>
          <span className="font-bold text-[var(--text-primary)]">{predictions.organ}</span>
        </div>
        {predictions.conditions.map((cond, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="font-semibold text-[var(--accent-light)] text-sm">{cond.label}</span>
              <span className="font-bold text-sm text-[var(--accent)]">
                {(cond.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <ConfidenceBar value={cond.confidence} />
            <div className="text-xs text-[var(--text-secondary)] text-right">
              Model: {cond.model}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
