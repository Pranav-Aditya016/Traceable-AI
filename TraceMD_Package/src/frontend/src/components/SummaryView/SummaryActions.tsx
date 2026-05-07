import React, { useState } from 'react';
import type { ReportData } from '../../types';

interface SummaryActionsProps {
  reportData: ReportData;
  onReset: () => void;
  onTabChange: (tab: string) => void;
}

export const SummaryActions: React.FC<SummaryActionsProps> = ({ reportData, onReset, onTabChange }) => {
  const [confirmed, setConfirmed] = useState(false);

  const handleExport = () => {
    if (!confirmed) return;
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report-${reportData.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)] space-y-4">
      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(e) => setConfirmed(e.target.checked)}
          className="mt-0.5 w-4 h-4 rounded border-[var(--border-color)] bg-[var(--bg-primary)] text-[var(--accent)] focus:ring-[var(--accent)] focus:ring-offset-0"
        />
        <span className="text-sm text-[var(--text-secondary)]">
          I confirm this is for research purposes only.
        </span>
      </label>
      <div className="grid grid-cols-3 gap-3">
        <button
          onClick={onReset}
          className="w-full px-3 py-2 border border-[var(--danger)]/50 rounded-md text-sm font-medium text-[var(--danger)] hover:bg-[var(--danger)]/10 transition disabled:opacity-50"
        >
          Reject
        </button>
        <button
          onClick={handleExport}
          disabled={!confirmed}
          className="w-full px-3 py-2 rounded-md text-sm font-medium text-white bg-[var(--accent)] hover:bg-[var(--accent-light)] disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          Approve + Export
        </button>
        <button
          onClick={() => onTabChange('NLE')}
          className="w-full px-3 py-2 border border-[var(--border-color)] rounded-md text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)] transition"
        >
          Open NLE
        </button>
      </div>
    </div>
  );
};
