import React, { useState } from 'react';
import type { ExtractedFields } from '../../types';

interface ExtractedDataProps {
  fileName: string;
  fields: ExtractedFields;
  ocrText: string;
}

const DataItem: React.FC<{ label: string; value: string | number | string[] }> = ({ label, value }) => (
  <div className="flex justify-between py-2.5 border-b border-[var(--border-color)]">
    <dt className="text-sm text-[var(--text-secondary)]">{label}</dt>
    <dd className="text-sm text-[var(--text-primary)] text-right font-semibold max-w-[60%]">
      {Array.isArray(value)
        ? value.length > 2
          ? value.slice(0, 2).join(', ') + '…'
          : value.join(', ')
        : value}
    </dd>
  </div>
);

export const ExtractedData: React.FC<ExtractedDataProps> = ({ fileName, fields, ocrText }) => {
  const [showOcr, setShowOcr] = useState(false);

  return (
    <div className="bg-[var(--bg-card)] p-4 rounded-lg border border-[var(--border-color)]">
      <h4 className="font-semibold text-[var(--text-primary)] mb-3">Extracted Data</h4>
      <dl>
        <DataItem label="File" value={fileName} />
        <DataItem label="Patient" value={fields.pseudonym} />
        <DataItem label="Age" value={fields.age} />
        <DataItem label="Sex" value={fields.sex} />
        <DataItem label="Symptoms" value={fields.symptoms} />
      </dl>
      <button
        onClick={() => setShowOcr(!showOcr)}
        className="text-[var(--accent-light)] hover:underline text-sm font-medium mt-4 block"
      >
        {showOcr ? 'Hide' : 'Show'} Full OCR Text
      </button>
      {showOcr && (
        <div className="mt-3 p-3 bg-[var(--bg-primary)] rounded-md text-xs text-[var(--text-secondary)] max-h-48 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-mono">{ocrText}</pre>
        </div>
      )}
    </div>
  );
};
