import React, { useState } from 'react';
import type { Localization } from '../../types';

interface LocalizationViewerProps {
  generatedImageB64: string;
  saliencyImageB64: string;
  maskImageB64: string;
  localization: Localization;
}

const ToggleSwitch: React.FC<{ label: string; checked: boolean; onChange: (c: boolean) => void }> = ({
  label, checked, onChange,
}) => (
  <label className="inline-flex items-center cursor-pointer gap-2">
    <span className="text-sm font-medium text-[var(--text-secondary)]">{label}</span>
    <span className="toggle-switch">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-slider" />
    </span>
  </label>
);

export const LocalizationViewer: React.FC<LocalizationViewerProps> = ({
  generatedImageB64, saliencyImageB64, maskImageB64, localization,
}) => {
  const [showMask, setShowMask] = useState(true);
  const [showSaliency, setShowSaliency] = useState(false);
  const [saliencyOpacity, setSaliencyOpacity] = useState(0.5);

  const maskPoints = localization.mask.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <div className="p-5 bg-[var(--bg-card)] rounded-lg border border-[var(--border-color)]">
      <div className="flex flex-wrap justify-between items-center mb-4 gap-4">
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">Localization & Provenance</h3>
        <div className="flex items-center space-x-4">
          <ToggleSwitch label="Mask" checked={showMask} onChange={setShowMask} />
          <div className="flex items-center gap-2">
            <ToggleSwitch label="Saliency" checked={showSaliency} onChange={setShowSaliency} />
            {showSaliency && (
              <input
                type="range" min="0" max="1" step="0.1" value={saliencyOpacity}
                onChange={(e) => setSaliencyOpacity(parseFloat(e.target.value))}
                className="w-20 h-1.5 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer"
                title={`Opacity: ${Math.round(saliencyOpacity * 100)}%`}
              />
            )}
          </div>
        </div>
      </div>

      <div className="relative w-full aspect-square bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center max-w-lg mx-auto">
        {generatedImageB64 ? (
          <>
            <img
              src={`data:image/png;base64,${generatedImageB64}`}
              alt="Synthetic medical image"
              className="object-contain w-full h-full"
            />
            {showSaliency && saliencyImageB64 && (
              <img
                src={`data:image/png;base64,${saliencyImageB64}`}
                alt="Saliency heatmap"
                className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
                style={{ opacity: saliencyOpacity }}
              />
            )}
            {showMask && (
              <svg viewBox="0 0 512 512" className="absolute inset-0 w-full h-full">
                <polygon
                  points={maskPoints}
                  className="fill-cyan-400/30 stroke-cyan-300"
                  strokeWidth="2"
                />
              </svg>
            )}
          </>
        ) : (
          <div className="text-[var(--text-secondary)] text-center">
            <p>Loading image…</p>
          </div>
        )}
        <div className="absolute top-2 left-2 bg-[var(--danger)] text-white text-[10px] font-bold px-3 py-1.5 rounded-full shadow-lg z-10 uppercase tracking-wider">
          Synthetic — Not for Diagnosis
        </div>
      </div>
    </div>
  );
};
