import React from 'react';
import type { AuditLogEntry } from '../../types';

interface AuditLogProps {
  entries: AuditLogEntry[];
}

export const AuditLog: React.FC<AuditLogProps> = ({ entries }) => {
  return (
    <div className="bg-white dark:bg-gray-800/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-3">Audit Log</h4>
      <div className="max-h-60 overflow-y-auto pr-2">
        <ol className="relative border-s border-gray-200 dark:border-gray-700">
          {entries.slice().reverse().map((entry, index) => (
            <li key={index} className="mb-4 ms-4">
              <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
              <time className="mb-1 text-xs font-normal leading-none text-gray-400 dark:text-gray-500">
                {new Date(entry.timestamp).toLocaleString()}
              </time>
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{entry.action}</h3>
              <p className="text-sm font-normal text-gray-500 dark:text-gray-400">by {entry.user}</p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
};
