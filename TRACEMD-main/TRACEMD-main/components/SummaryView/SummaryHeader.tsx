import React from 'react';

interface SummaryHeaderProps {
  reportId: string;
  pseudonym: string;
  createdAt: string;
  onReset: () => void;
}

export const SummaryHeader: React.FC<SummaryHeaderProps> = ({ reportId, pseudonym, createdAt, onReset }) => {
  return (
    <div className="flex flex-wrap justify-between items-center gap-4 pb-4 border-b border-gray-200 dark:border-gray-700">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Analysis Summary</h2>
        <div className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-4 mt-1">
          <span>Report ID: <span className="font-mono">{reportId}</span></span>
          <span>Patient: <span className="font-semibold">{pseudonym}</span></span>
          <span>Created: <span className="font-semibold">{new Date(createdAt).toLocaleString()}</span></span>
        </div>
      </div>
       <div className="flex items-center gap-4">
        <span className="text-xs font-semibold text-red-600 border border-red-500 rounded-full px-3 py-1">
            SYNTHETIC — NOT FOR DIAGNOSIS
        </span>
        <button
            onClick={onReset}
            className="bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-indigo-700 transition duration-300"
            >
            New Analysis
        </button>
       </div>
    </div>
  );
};
