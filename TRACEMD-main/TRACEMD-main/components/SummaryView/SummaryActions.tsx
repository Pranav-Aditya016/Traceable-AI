import React, { useState } from 'react';
import type { ReportData } from '../../types';
import { Spinner } from '../Spinner';

interface SummaryActionsProps {
  status: ReportData['status'];
  onUpdateStatus: (newStatus: 'approved' | 'rejected') => void;
  onRegenerate: () => Promise<void>;
  isRegenerating: boolean;
}

export const SummaryActions: React.FC<SummaryActionsProps> = ({ status, onUpdateStatus, onRegenerate, isRegenerating }) => {
    const [confirmation, setConfirmation] = useState(false);
    
    const isActionable = status === 'generated';

    if (!isActionable) {
        return (
             <div className={`p-4 rounded-lg text-center font-semibold capitalize ${status === 'approved' ? 'bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-300' : 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-800 dark:text-yellow-300'}`}>
                Result has been marked as {status}.
            </div>
        )
    }

    return (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 space-y-4">
             <div className="flex items-start">
                <div className="flex items-center h-5">
                <input
                    id="confirmation"
                    type="checkbox"
                    checked={confirmation}
                    onChange={(e) => setConfirmation(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 bg-gray-100 border-gray-300 rounded focus:ring-indigo-500 dark:focus:ring-indigo-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                </div>
                <div className="ms-3 text-sm">
                <label htmlFor="confirmation" className="font-medium text-gray-700 dark:text-gray-300">
                    I confirm this is for research purposes only.
                </label>
                </div>
            </div>
            <div className="grid grid-cols-3 gap-3">
                <button
                    onClick={() => onUpdateStatus('rejected')}
                    disabled={!confirmation}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                    Reject
                </button>
                <button
                    onClick={() => onRegenerate()}
                    disabled={!confirmation || isRegenerating}
                    className="w-full flex justify-center items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition"
                >
                    {isRegenerating && <Spinner />}
                    Regenerate
                </button>
                <button
                    onClick={() => onUpdateStatus('approved')}
                    disabled={!confirmation}
                    className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed transition"
                >
                    Approve
                </button>
            </div>
        </div>
    );
};
