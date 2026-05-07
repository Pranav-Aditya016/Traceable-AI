import React from 'react';

interface SummaryHeaderProps {
  reportId: string;
  pseudonym: string;
  createdAt: string;
  onReset: () => void;
}

export const SummaryHeader: React.FC<SummaryHeaderProps> = ({ reportId, pseudonym, createdAt, onReset }) => {
  return (
    <div className="flex flex-wrap justify-between items-center gap-4 pb-4 border-b border-[var(--border-color)]">
      <div>
        <h2 className="text-2xl font-bold text-[var(--text-primary)]">Results Dashboard</h2>
        <div className="text-sm text-[var(--text-secondary)] flex flex-wrap items-center gap-4 mt-1">
          <span>Report ID: <span className="font-mono text-[var(--text-primary)]">{reportId}</span></span>
          <span>Patient: <span className="font-semibold text-[var(--text-primary)]">{pseudonym}</span></span>
          <span>Created: <span className="font-semibold text-[var(--text-primary)]">{new Date(createdAt).toLocaleString()}</span></span>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-xs font-semibold text-[var(--danger)] border border-[var(--danger)] rounded-full px-3 py-1 uppercase tracking-wide">
          Synthetic — Not for Diagnosis
        </span>
        <button
          onClick={onReset}
          className="bg-[var(--accent)] text-white font-bold py-2 px-5 rounded-lg hover:bg-[var(--accent-light)] transition duration-300"
        >
          New Analysis
        </button>
      </div>
    </div>
  );
};
