import React, { useState } from 'react';
import type { Localization } from '../../types';
import { Spinner } from '../Spinner';

interface LocalizationViewerProps {
  imageUrl: string | null;
  saliencyUrl: string | null;
  localization: Localization;
}

export const LocalizationViewer: React.FC<LocalizationViewerProps> = ({ imageUrl, saliencyUrl, localization }) => {
  const [showMask, setShowMask] = useState(true);
  const [showSaliency, setShowSaliency] = useState(false);
  const [saliencyOpacity, setSaliencyOpacity] = useState(0.5);

  const maskPoints = localization.mask.map(p => `${p.x},${p.y}`).join(' ');

  return (
    <div className="p-4 bg-white dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex flex-wrap justify-between items-center mb-4 gap-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Localization & Provenance</h3>
            <div className="flex items-center space-x-4">
                <ToggleSwitch label="Mask" checked={showMask} onChange={setShowMask} />
                <div className="flex items-center">
                    <ToggleSwitch label="Saliency" checked={showSaliency} onChange={setShowSaliency} />
                    {showSaliency && (
                       <div className="flex items-center ml-2">
                         <input
                           type="range"
                           min="0"
                           max="1"
                           step="0.1"
                           value={saliencyOpacity}
                           onChange={(e) => setSaliencyOpacity(parseFloat(e.target.value))}
                           className="w-24 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                           title={`Opacity: ${Math.round(saliencyOpacity * 100)}%`}
                         />
                       </div>
                    )}
                </div>
            </div>
        </div>

        <div className="relative w-full aspect-square bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center">
        {imageUrl ? (
            <>
            <img src={imageUrl} alt="Synthetic medical image" className="object-contain w-full h-full" />
            
            {showSaliency && saliencyUrl && (
                 <img 
                    src={saliencyUrl} 
                    alt="Saliency heatmap" 
                    className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
                    style={{ opacity: saliencyOpacity }}
                 />
            )}

            {showMask && (
                <svg viewBox="0 0 512 512" className="absolute inset-0 w-full h-full">
                    <polygon
                        points={maskPoints}
                        className="fill-cyan-400/50 stroke-cyan-300 stroke-2"
                    />
                </svg>
            )}
            </>
        ) : (
            <div className="text-white text-center">
                <Spinner />
                <p className="mt-2">Loading image...</p>
            </div>
        )}
        
        <div className="absolute top-2 left-2 bg-red-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg z-10">
            SYNTHETIC — NOT FOR DIAGNOSIS
        </div>
        </div>
    </div>
  );
};

const ToggleSwitch: React.FC<{label: string, checked: boolean, onChange: (c: boolean) => void}> = ({label, checked, onChange}) => (
    <label className="inline-flex items-center cursor-pointer">
        <span className="mr-3 text-sm font-medium text-gray-900 dark:text-gray-300">{label}</span>
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="sr-only peer" />
        <div className="relative w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
    </label>
);