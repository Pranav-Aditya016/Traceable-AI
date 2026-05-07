import React, { useState } from 'react';
import type { ExtractedFields } from '../../types';

interface OcrDetailsProps {
  fileName: string;
  fields: ExtractedFields;
  ocrText: string;
}

const DataItem: React.FC<{ label: string; value: string | number | string[] }> = ({ label, value }) => (
    <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-700">
        <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</dt>
        <dd className="text-sm text-gray-900 dark:text-gray-100 text-right font-semibold">{Array.isArray(value) ? value.join(', ') : value}</dd>
    </div>
);

export const OcrDetails: React.FC<OcrDetailsProps> = ({ fileName, fields, ocrText }) => {
  const [showOcrText, setShowOcrText] = useState(false);

  return (
    <div className="bg-white dark:bg-gray-800/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-3">Extracted Data</h4>
      <dl>
        <DataItem label="File Name" value={fileName} />
        <DataItem label="Patient" value={fields.pseudonym} />
        <DataItem label="Age" value={fields.age} />
        <DataItem label="Sex" value={fields.sex} />
        <DataItem label="Symptoms" value={fields.symptoms} />
      </dl>
      <button
        onClick={() => setShowOcrText(!showOcrText)}
        className="text-indigo-600 dark:text-indigo-400 hover:underline text-sm font-medium mt-4"
      >
        {showOcrText ? 'Hide' : 'Show'} Full OCR Text
      </button>
      {showOcrText && (
        <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-900 rounded-md text-xs text-gray-600 dark:text-gray-300 max-h-48 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-mono">{ocrText}</pre>
        </div>
      )}
    </div>
  );
};
