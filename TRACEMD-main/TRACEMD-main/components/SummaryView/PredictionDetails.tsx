import React from 'react';
import type { ReportData } from '../../types';

interface PredictionDetailsProps {
  predictions: ReportData['predictions'];
}

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => {
    const percentage = Math.round(value * 100);
    let color = 'bg-green-500';
    if (percentage < 75) color = 'bg-yellow-500';
    if (percentage < 50) color = 'bg-red-500';

    return (
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
            <div className={`${color} h-2.5 rounded-full`} style={{ width: `${percentage}%` }}></div>
        </div>
    );
};

export const PredictionDetails: React.FC<PredictionDetailsProps> = ({ predictions }) => {
  return (
    <div className="bg-white dark:bg-gray-800/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-3">AI Prediction</h4>
      <div className="space-y-4">
        <div className="flex justify-between text-sm">
            <span className="font-medium text-gray-500 dark:text-gray-400">Predicted Organ:</span>
            <span className="font-bold text-gray-900 dark:text-gray-100">{predictions.organ}</span>
        </div>
        {predictions.conditions.map((cond, index) => (
            <div key={index} className="space-y-2">
                <div className="flex justify-between items-center">
                    <span className="font-semibold text-indigo-800 dark:text-indigo-300">{cond.label}</span>
                    <span className="font-bold text-sm text-gray-800 dark:text-gray-200">{(cond.confidence * 100).toFixed(1)}%</span>
                </div>
                <ConfidenceBar value={cond.confidence} />
                <div className="text-xs text-gray-500 dark:text-gray-400 text-right">Model: {cond.model}</div>
            </div>
        ))}
      </div>
    </div>
  );
};
