import React, { useState } from 'react';
import type { ReportData } from '../../types';
import { useReportSummary } from '../../hooks/useReportSummary';
import { SummaryHeader } from './SummaryHeader';
import { OverviewCard } from './OverviewCard';
import { OcrDetails } from './OcrDetails';
import { PredictionDetails } from './PredictionDetails';
import { LocalizationViewer } from './LocalizationViewer';
import { XaiVisualizations } from './XaiVisualizations';
import { AuditLog } from './AuditLog';
import { SummaryActions } from './SummaryActions';
import { Tabs } from '../Tabs';
import { NLETab } from './NLETab';
import { PipelineExplanation } from './PipelineExplanation';

interface SummaryViewProps {
  initialData: ReportData;
  onReset: () => void;
}

export const SummaryView: React.FC<SummaryViewProps> = ({ initialData, onReset }) => {
  const { data, loading, error, updateStatus, regenerate, saveExplanation } = useReportSummary(initialData);
  const [activeTab, setActiveTab] = useState('Analysis Summary');

  if (!data) return null;

  return (
    <div className="space-y-8">
      <SummaryHeader
        reportId={data.id}
        createdAt={data.createdAt}
        pseudonym={data.fields.pseudonym}
        onReset={onReset}
      />

      <Tabs
        tabs={['Analysis Summary', 'Natural Language Explanation']}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {activeTab === 'Analysis Summary' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="xl:col-span-2 space-y-6">
            <OverviewCard xai={data.xai} predictions={data.predictions} />
            <LocalizationViewer
              imageUrl={data.generatedImageUrl}
              saliencyUrl={data.xai.saliencyImageUrl}
              localization={data.predictions.localization}
            />
            <XaiVisualizations xai={data.xai} />
            <PipelineExplanation reportData={data} />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <SummaryActions
              status={data.status}
              onUpdateStatus={updateStatus}
              onRegenerate={regenerate}
              isRegenerating={loading}
            />
            <OcrDetails
              fileName={data.fileName}
              fields={data.fields}
              ocrText={data.ocrText}
            />
            <PredictionDetails predictions={data.predictions} />
            <AuditLog entries={data.auditLog} />
          </div>
        </div>
      )}

      {activeTab === 'Natural Language Explanation' && (
        <NLETab reportData={data} onSave={saveExplanation} />
      )}

       {error && <p className="text-red-500">Error during regeneration: {error}</p>}
    </div>
  );
};