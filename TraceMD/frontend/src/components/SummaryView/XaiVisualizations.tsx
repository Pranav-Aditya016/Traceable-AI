import React, { useState, useMemo } from 'react';
import type { XAI, ShapEntry } from '../../types';

interface XaiVisualizationsProps {
  xai: XAI;
}

export const XaiVisualizations: React.FC<XaiVisualizationsProps> = ({ xai }) => {
  const [sortConfig, setSortConfig] = useState<{ key: keyof ShapEntry; direction: 'asc' | 'desc' } | null>(null);

  const sortedShapTable = useMemo(() => {
    const items = [...xai.shap_table];
    if (sortConfig) {
      items.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
        if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return items;
  }, [xai.shap_table, sortConfig]);

  const requestSort = (key: keyof ShapEntry) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig?.key === key && sortConfig.direction === 'asc') direction = 'desc';
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key: keyof ShapEntry) => {
    if (!sortConfig || sortConfig.key !== key) return <span className="text-[var(--text-secondary)] opacity-40">↕</span>;
    return sortConfig.direction === 'asc'
      ? <span className="text-[var(--accent)]">▲</span>
      : <span className="text-[var(--accent)]">▼</span>;
  };

  const maxAbsImpact = useMemo(() => {
    if (!xai.shap_table.length) return 1;
    return Math.max(...xai.shap_table.map((s) => Math.abs(s.impact)), 0.01);
  }, [xai.shap_table]);

  return (
    <div className="p-5 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
      <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">Explainable AI (XAI) Artifacts</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Text Features */}
        <div>
          <h4 className="font-semibold text-[var(--text-primary)] mb-1">Top Text Features</h4>
          <p className="text-xs text-[var(--text-secondary)] mb-3">Keywords from the input that most influenced the prediction.</p>
          <ul className="space-y-2">
            {xai.text_features.map((item, i) => (
              <li key={i} className="flex justify-between items-center text-sm p-2.5 bg-[var(--bg-primary)] rounded-md">
                <span className="font-mono text-[var(--accent-light)]">{item.feature}</span>
                <span className={`font-semibold ${item.weight > 0 ? 'text-[var(--success)]' : 'text-[var(--danger)]'}`}>
                  {item.weight.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* SHAP Table */}
        <div>
          <div className="flex items-center mb-1 gap-2">
            <h4 className="font-semibold text-[var(--text-primary)]">SHAP Values</h4>
            <div className="relative group">
              <svg className="w-4 h-4 text-[var(--text-secondary)]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="absolute bottom-full z-20 mb-2 w-64 p-2.5 bg-gray-800 text-white text-xs rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none left-1/2 -translate-x-1/2">
                SHAP (SHapley Additive exPlanations) values show the impact of each feature. Positive values push the prediction higher, negative values push it lower.
              </div>
            </div>
          </div>
          <p className="text-xs text-[var(--text-secondary)] mb-3">The impact of each feature value on the model output.</p>
          <div className="max-h-52 overflow-y-auto border border-[var(--border-color)] rounded-md">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-[var(--text-secondary)] uppercase bg-[var(--bg-primary)] sticky top-0">
                <tr>
                  <th className="px-3 py-2">
                    <button onClick={() => requestSort('feature')} className="flex items-center gap-1 hover:text-[var(--accent)]">Feature {getSortIcon('feature')}</button>
                  </th>
                  <th className="px-3 py-2">
                    <button onClick={() => requestSort('value')} className="flex items-center gap-1 hover:text-[var(--accent)]">Value {getSortIcon('value')}</button>
                  </th>
                  <th className="px-3 py-2">
                    <button onClick={() => requestSort('impact')} className="flex items-center gap-1 hover:text-[var(--accent)]">Impact {getSortIcon('impact')}</button>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedShapTable.map((item, i) => (
                  <tr key={i} className="border-b border-[var(--border-color)] hover:bg-[var(--bg-card-hover)]">
                    <td className="px-3 py-2 font-medium text-[var(--text-primary)]">{item.feature}</td>
                    <td className="px-3 py-2 font-mono text-[var(--text-secondary)]">{item.value}</td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-3 bg-[var(--bg-primary)] rounded overflow-hidden relative">
                          <div
                            className={item.impact >= 0 ? 'shap-bar-positive' : 'shap-bar-negative'}
                            style={{
                              width: `${(Math.abs(item.impact) / maxAbsImpact) * 100}%`,
                              height: '100%',
                              ...(item.impact < 0 ? { marginLeft: 'auto' } : {}),
                            }}
                          />
                        </div>
                        <span className={`text-xs font-semibold ${item.impact >= 0 ? 'text-[var(--accent-light)]' : 'text-[var(--danger)]'}`}>
                          {item.impact > 0 ? '+' : ''}{item.impact.toFixed(3)}
                        </span>
                      </div>
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
