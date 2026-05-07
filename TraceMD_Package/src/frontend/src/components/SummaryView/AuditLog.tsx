import React from 'react';
import type { AuditLogEntry } from '../../types';

interface AuditLogProps {
  entries: AuditLogEntry[];
}

export const AuditLog: React.FC<AuditLogProps> = ({ entries }) => {
  return (
    <div className="bg-[var(--bg-card)] p-4 rounded-lg border border-[var(--border-color)]">
      <h4 className="font-semibold text-[var(--text-primary)] mb-3">Audit Log</h4>
      <div className="max-h-60 overflow-y-auto pr-2">
        <ol className="relative border-s border-[var(--border-color)] ml-2">
          {entries.slice().reverse().map((entry, i) => (
            <li key={i} className="mb-5 ms-4">
              <div className="absolute w-2.5 h-2.5 bg-[var(--accent)] rounded-full mt-1.5 -start-[5.5px] border-2 border-[var(--bg-card)]" />
              <time className="mb-1 text-xs text-[var(--text-secondary)] block">
                {new Date(entry.timestamp).toLocaleString()}
              </time>
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">{entry.action}</h3>
              <p className="text-xs text-[var(--text-secondary)]">by {entry.by}</p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
};
