import { motion } from 'framer-motion';
import type { PipelineStep } from '../types';

interface PipelineTrackerProps {
  steps: PipelineStep[];
}

const statusIcons: Record<string, string> = {
  pending: '○',
  running: '◌',
  complete: '✓',
  error: '✗',
};

export function PipelineTracker({ steps }: PipelineTrackerProps) {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-[var(--text-primary)] mb-4">Pipeline Progress</h2>
      {steps.map((step, i) => (
        <motion.div
          key={step.id}
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: i * 0.08 }}
          className={`flex items-start gap-4 p-4 rounded-xl border transition-all ${
            step.status === 'running'
              ? 'bg-[var(--accent)]/5 border-[var(--accent)]/40'
              : step.status === 'complete'
              ? 'bg-[var(--success)]/5 border-[var(--success)]/30'
              : step.status === 'error'
              ? 'bg-[var(--danger)]/5 border-[var(--danger)]/30'
              : 'bg-[var(--bg-card)] border-[var(--border-color)]'
          }`}
        >
          <div
            className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
              step.status === 'running'
                ? 'bg-[var(--accent)] text-white animate-pulse'
                : step.status === 'complete'
                ? 'bg-[var(--success)] text-white'
                : step.status === 'error'
                ? 'bg-[var(--danger)] text-white'
                : 'bg-[var(--bg-primary)] text-[var(--text-secondary)] border border-[var(--border-color)]'
            }`}
          >
            {statusIcons[step.status] || step.id}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-[var(--accent)] uppercase tracking-wide">Stage {step.id}</span>
              <span className="text-sm font-semibold text-[var(--text-primary)]">{step.model}</span>
            </div>
            {step.result && (
              <p className="text-sm text-[var(--text-secondary)] truncate">{step.result}</p>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
