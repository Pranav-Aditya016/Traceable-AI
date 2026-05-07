import React, { useState, useMemo } from 'react';
import type { XAI, ShapValue } from '../../types';

interface XaiVisualizationsProps {
  xai: XAI;
}

export const XaiVisualizations: React.FC<XaiVisualizationsProps> = ({ xai }) => {
  const [sortConfig, setSortConfig] = useState<{ key: keyof ShapValue; direction: 'ascending' | 'descending' } | null>(null);

  const sortedShapTable = useMemo(() => {
    let sortableItems = [...xai.shap_table];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [xai.shap_table, sortConfig]);

  const requestSort = (key: keyof ShapValue) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };
  
  const getSortIcon = (key: keyof ShapValue) => {
    if (!sortConfig || sortConfig.key !== key) {
      return <span className="text-gray-400">↕</span>;
    }
    return sortConfig.direction === 'ascending' ? <span className="text-indigo-500">▲</span> : <span className="text-indigo-500">▼</span>;
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-4">Explainable AI (XAI) Artifacts</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-2">Top Text Features</h4>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Keywords from the report that most influenced the prediction.</p>
          <ul className="space-y-2">
            {xai.text_features.map((item, index) => (
              <li key={index} className="flex justify-between items-center text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded-md">
                <span className="font-mono text-indigo-700 dark:text-indigo-300">{item.feature}</span>
                <span className={`font-semibold ${item.weight > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {item.weight.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div>
            <div className="flex items-center mb-2">
                <h4 className="font-semibold text-gray-700 dark:text-gray-200">SHAP Values</h4>
                <div className="relative group ml-2">
                    <svg className="w-4 h-4 text-gray-400 dark:text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="absolute bottom-full z-10 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                        SHAP (SHapley Additive exPlanations) values show the impact of each feature on the model's prediction. Positive values push the prediction higher, negative values push it lower.
                        <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-gray-800"></div>
                    </div>
                </div>
            </div>
           <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">The impact of each feature value on the model output.</p>
          <div className="max-h-48 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-md">
            <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-700 dark:text-gray-300 uppercase bg-gray-100 dark:bg-gray-700 sticky top-0">
                    <tr>
                        <th scope="col" className="px-4 py-2">
                            <button onClick={() => requestSort('feature')} className="flex items-center gap-1 hover:text-indigo-500">Feature {getSortIcon('feature')}</button>
                        </th>
                        <th scope="col" className="px-4 py-2">
                            <button onClick={() => requestSort('value')} className="flex items-center gap-1 hover:text-indigo-500">Value {getSortIcon('value')}</button>
                        </th>
                        <th scope="col" className="px-4 py-2 text-right">
                            <button onClick={() => requestSort('shap')} className="flex items-center gap-1 w-full justify-end hover:text-indigo-500">Impact {getSortIcon('shap')}</button>
                        </th>
                    </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800">
                {sortedShapTable.map((item, index) => (
                    <tr key={index} className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-2 font-medium">{item.feature}</td>
                        <td className="px-4 py-2 font-mono">{item.value}</td>
                        <td className={`px-4 py-2 font-semibold text-right ${item.shap > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {item.shap.toFixed(3)}
                        </td>
                    </tr>
                ))}
                </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};