
import React from 'react';
import type { PipelineStatus as Status } from '../types';
import { Spinner } from './Spinner';

interface PipelineStatusProps {
  status: Status;
}

const statusConfig = {
  ocr: { text: 'Analyzing Report (OCR & Field Extraction)', color: 'blue', running: true },
  predicting: { text: 'Predicting Condition & Organ', color: 'blue', running: true },
  generating: { text: 'Generating Synthetic Image', color: 'blue', running: true },
  complete: { text: 'Analysis Complete', color: 'green', running: false },
  error: { text: 'An Error Occurred', color: 'red', running: false },
  idle: { text: 'Awaiting Upload', color: 'gray', running: false },
};

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ status }) => {
  const { text, color, running } = statusConfig[status] || statusConfig.idle;
  
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-800',
    green: 'bg-green-100 text-green-800',
    red: 'bg-red-100 text-red-800',
    gray: 'bg-gray-100 text-gray-800'
  }[color];

  return (
    <div className={`flex items-center p-4 rounded-lg ${colorClasses}`}>
      {running && <Spinner />}
      {!running && status === 'complete' && (
        <svg className="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path></svg>
      )}
      {!running && status === 'error' && (
        <svg className="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"></path></svg>
      )}
      <span className="font-medium">{text}</span>
    </div>
  );
};
