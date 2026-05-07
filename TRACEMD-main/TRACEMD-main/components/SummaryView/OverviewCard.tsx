import React from 'react';
import type { XAI, ReportData } from '../../types';

interface OverviewCardProps {
    xai: XAI;
    predictions: ReportData['predictions'];
}

export const OverviewCard: React.FC<OverviewCardProps> = ({ xai, predictions }) => {
  const primaryCondition = predictions.conditions[0];
  const confidence = primaryCondition ? (primaryCondition.confidence * 100).toFixed(0) : 'N/A';

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-start">
            <div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">AI Summary</h3>
                <p className="mt-2 text-gray-600 dark:text-gray-300 leading-relaxed">
                    {xai.naturalLanguageSummary}
                </p>
            </div>
            <div className="text-center ml-6 flex-shrink-0">
                <div className={`relative h-20 w-20 rounded-full flex items-center justify-center text-white font-bold text-2xl bg-gradient-to-br from-indigo-500 to-purple-600`}>
                    {confidence}%
                </div>
                <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mt-2">Overall Confidence</p>
            </div>
        </div>
    </div>
  );
};
