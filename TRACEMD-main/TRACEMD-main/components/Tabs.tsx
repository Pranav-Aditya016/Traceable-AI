import React from 'react';

interface TabsProps {
  tabs: string[];
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, setActiveTab }) => {
  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <nav className="-mb-px flex space-x-8" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`${
              activeTab === tab
                ? 'border-indigo-500 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:border-gray-500'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors`}
            aria-current={activeTab === tab ? 'page' : undefined}
          >
            {tab}
          </button>
        ))}
      </nav>
    </div>
  );
};
