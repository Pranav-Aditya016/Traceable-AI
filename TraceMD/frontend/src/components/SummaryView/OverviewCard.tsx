import React from 'react';
import type { XAI, ReportData } from '../../types';

interface OverviewCardProps {
  xai: XAI;
  predictions: ReportData['predictions'];
}

const ConfidenceRing: React.FC<{ value: number }> = ({ value }) => {
  const percentage = Math.round(value * 100);
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value * circumference);

  return (
    <div className="confidence-ring" style={{ width: 88, height: 88 }}>
      <svg width="88" height="88">
        <circle cx="44" cy="44" r={radius} fill="none" stroke="rgba(99,102,241,0.15)" strokeWidth="6" />
        <circle
          cx="44" cy="44" r={radius} fill="none"
          stroke="url(#confidence-gradient)" strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
        <defs>
          <linearGradient id="confidence-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
        </defs>
      </svg>
      <span className="ring-text">{percentage}%</span>
    </div>
  );
};

export const OverviewCard: React.FC<OverviewCardProps> = ({ xai, predictions }) => {
  const overallConfidence = predictions.overall_confidence ?? (predictions.conditions[0]?.confidence || 0);

  return (
    <div className="p-6 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
      <div className="flex justify-between items-start">
        <div className="flex-1 mr-6">
          <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">AI Summary</h3>
          <p className="text-[var(--text-secondary)] leading-relaxed text-sm">
            {xai.naturalLanguageSummary || 'Summary will be generated after pipeline completion.'}
          </p>
        </div>
        <div className="text-center flex-shrink-0">
          <ConfidenceRing value={overallConfidence} />
          <p className="text-xs font-semibold text-[var(--text-secondary)] mt-2">Overall Confidence</p>
        </div>
      </div>
    </div>
  );
};
