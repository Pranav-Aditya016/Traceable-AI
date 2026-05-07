import { useCallback, useState } from 'react';
import { motion } from 'framer-motion';

interface UploadZoneProps {
  onUpload: (file: File) => void;
  detectedBadge: string;
}

export function UploadZone({ onUpload, detectedBadge }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = () => setPreview(reader.result as string);
        reader.readAsDataURL(file);
      } else {
        setPreview(null);
      }
      onUpload(file);
    },
    [onUpload]
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className={`relative p-8 rounded-2xl border-2 border-dashed transition-all ${
        isDragging
          ? 'border-[var(--accent)] bg-[var(--accent)]/5'
          : 'border-[var(--border-color)] bg-[var(--bg-card)]'
      }`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
    >
      <div className="text-center">
        <div className="mb-4">
          <svg
            className="mx-auto w-12 h-12 text-[var(--accent)]"
            xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 3 3 0 013.438 3.17A3.75 3.75 0 0118 19.5H6.75z"
            />
          </svg>
        </div>
        <p className="text-[var(--text-primary)] font-semibold mb-2">
          Drop your file here, or{' '}
          <label className="text-[var(--accent-light)] cursor-pointer hover:underline">
            browse
            <input
              type="file"
              accept="image/*,.pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
          </label>
        </p>
        <p className="text-sm text-[var(--text-secondary)]">
          Supports JPG, PNG, and PDF · Max 20MB
        </p>
      </div>

      {(preview || fileName) && (
        <div className="mt-6 flex items-center gap-4 p-3 bg-[var(--bg-primary)] rounded-lg">
          {preview && (
            <img src={preview} alt="Preview" className="w-16 h-16 object-cover rounded-lg border border-[var(--border-color)]" />
          )}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-[var(--text-primary)] truncate">{fileName}</p>
            {detectedBadge && (
              <span className="inline-block mt-1 text-xs font-semibold text-[var(--medical-cyan)] bg-[var(--medical-cyan)]/10 px-2 py-0.5 rounded">
                {detectedBadge}
              </span>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
