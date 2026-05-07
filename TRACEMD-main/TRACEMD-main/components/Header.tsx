
import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <svg className="h-8 w-8 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M12 6V3m0 18v-3M5.636 5.636l-1.414-1.414M19.778 19.778l-1.414-1.414M4.222 19.778l1.414-1.414M18.364 5.636l1.414-1.414" />
          </svg>
          <h1 className="text-xl font-bold text-gray-800">Medical AI Pipeline</h1>
        </div>
        <span className="text-xs font-semibold text-red-600 border border-red-500 rounded-full px-3 py-1">
          RESEARCH PROTOTYPE
        </span>
      </div>
    </header>
  );
};
