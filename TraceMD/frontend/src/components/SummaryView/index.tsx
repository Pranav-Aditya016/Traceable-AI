import React, { useState } from 'react';
import type { ReportData } from '../../types';
import { Tabs } from './Tabs';
import { SummaryHeader } from './SummaryHeader';
import { OverviewCard } from './OverviewCard';
import { LocalizationViewer } from './LocalizationViewer';
import { XaiVisualizations } from './XaiVisualizations';
import { PipelineExplanation } from './PipelineExplanation';
import { SummaryActions } from './SummaryActions';
import { ExtractedData } from './ExtractedData';
import { PredictionDetails } from './PredictionDetails';
import { AuditLog } from './AuditLog';
import { NLExplanationTab } from './NLExplanationTab';

interface SummaryViewProps {
  data: ReportData;
  onReset: () => void;
}

const DASHBOARD_TABS = ['Overview', 'OCR', 'Predictions', 'Localization', 'XAI', 'NLE'] as const;
type DashboardTab = typeof DASHBOARD_TABS[number];

export const SummaryView: React.FC<SummaryViewProps> = ({ data, onReset }) => {
  const [activeTab, setActiveTab] = useState<DashboardTab>('Overview');

  const provenance = data.xai.provenance || { generator: 'N/A', xai_model: 'N/A' };
  const reasoningTrace = data.xai.reasoningTrace || [];
  const packageVersions = provenance.package_versions || {};

  return (
    <div className="space-y-6">
      <SummaryHeader
        reportId={data.id}
        createdAt={data.createdAt}
        pseudonym={data.fields.pseudonym}
        onReset={onReset}
      />

      <Tabs
        tabs={[...DASHBOARD_TABS]}
        activeTab={activeTab}
        setActiveTab={(tab) => setActiveTab(tab as DashboardTab)}
      />

      {activeTab === 'Overview' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 space-y-6">
            <OverviewCard xai={data.xai} predictions={data.predictions} />
            <PipelineExplanation reportData={data} />
          </div>

          <div className="space-y-6">
            <SummaryActions
              reportData={data}
              onReset={onReset}
              onTabChange={(tab) => setActiveTab(tab as DashboardTab)}
            />
            <AuditLog entries={data.auditLog} />
          </div>
        </div>
      )}

      {activeTab === 'OCR' && (
        <div className="max-w-5xl">
          <ExtractedData
            fileName={data.fileName}
            fields={data.fields}
            ocrText={data.ocrText}
          />
        </div>
      )}

      {activeTab === 'Predictions' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-1">
            <PredictionDetails predictions={data.predictions} />
          </div>
          <div className="xl:col-span-2 p-5 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
            <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-3">Reasoning Trace</h3>
            {reasoningTrace.length > 0 ? (
              <ul className="space-y-2 list-disc pl-5 text-sm text-[var(--text-secondary)]">
                {reasoningTrace.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-[var(--text-secondary)]">Reasoning trace is not available for this report.</p>
            )}
          </div>
        </div>
      )}

      {activeTab === 'Localization' && (
        <LocalizationViewer
          generatedImageB64={data.generatedImageB64}
          saliencyImageB64={data.xai.saliencyImageB64}
          maskImageB64={data.xai.maskImageB64}
          localization={data.predictions.localization}
        />
      )}

      {activeTab === 'XAI' && (
        <div className="space-y-6">
          <XaiVisualizations xai={data.xai} />

          <div className="p-5 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
            <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-3">Provenance & Reproducibility</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="bg-[var(--bg-primary)] rounded-md p-3 border border-[var(--border-color)]">
                <p className="text-[var(--text-secondary)]">Generator</p>
                <p className="font-semibold text-[var(--text-primary)]">{provenance.generator}</p>
              </div>
              <div className="bg-[var(--bg-primary)] rounded-md p-3 border border-[var(--border-color)]">
                <p className="text-[var(--text-secondary)]">XAI Model</p>
                <p className="font-semibold text-[var(--text-primary)]">{provenance.xai_model}</p>
              </div>
              <div className="bg-[var(--bg-primary)] rounded-md p-3 border border-[var(--border-color)]">
                <p className="text-[var(--text-secondary)]">OCR Model Selection</p>
                <p className="font-semibold text-[var(--text-primary)]">{provenance.ocr_selected_model || 'N/A'}</p>
              </div>
              <div className="bg-[var(--bg-primary)] rounded-md p-3 border border-[var(--border-color)]">
                <p className="text-[var(--text-secondary)]">Generation Seed</p>
                <p className="font-semibold text-[var(--text-primary)]">{provenance.seed ?? 'N/A'}</p>
              </div>
            </div>

            {Object.keys(packageVersions).length > 0 && (
              <div className="mt-4 border border-[var(--border-color)] rounded-md overflow-hidden">
                <div className="px-3 py-2 bg-[var(--bg-primary)] text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
                  Package Versions
                </div>
                <div className="divide-y divide-[var(--border-color)]">
                  {Object.entries(packageVersions).map(([name, pkgVersion]) => (
                    <div key={name} className="px-3 py-2 flex items-center justify-between text-sm">
                      <span className="text-[var(--text-secondary)]">{name}</span>
                      <span className="font-mono text-[var(--text-primary)]">{pkgVersion}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'NLE' && (
        <NLExplanationTab reportData={data} onReset={onReset} />
      )}
    </div>
  );
};
