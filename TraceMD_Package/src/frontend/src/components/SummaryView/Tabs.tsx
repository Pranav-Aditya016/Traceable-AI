import React from 'react';

interface TabsProps {
  tabs: string[];
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, setActiveTab }) => {
  return (
    <div className="border-b border-[var(--border-color)] overflow-x-auto">
      <nav className="-mb-px flex gap-6 min-w-max" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`${
              activeTab === tab
                ? 'border-[var(--accent)] text-[var(--accent-light)]'
                : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-color)]'
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
