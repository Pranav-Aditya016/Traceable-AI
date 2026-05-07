import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, FileImage, Send, AlertCircle } from 'lucide-react'

interface Props {
  files: File[]
  onFilesChange: (files: File[]) => void
  clinicalText: string
  onClinicalTextChange: (v: string) => void
  patientName: string
  onPatientNameChange: (v: string) => void
  patientAge: string
  onPatientAgeChange: (v: string) => void
  patientSex: string
  onPatientSexChange: (v: string) => void
  onAnalyze: () => void
  disabled: boolean
  error: string
}

export default function UploadPanel({
  files, onFilesChange,
  clinicalText, onClinicalTextChange,
  patientName, onPatientNameChange,
  patientAge, onPatientAgeChange,
  patientSex, onPatientSexChange,
  onAnalyze, disabled, error,
}: Props) {

  const onDrop = useCallback((accepted: File[]) => {
    onFilesChange([...files, ...accepted])
  }, [files, onFilesChange])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm'],
    },
    multiple: true,
  })

  const removeFile = (idx: number) => {
    onFilesChange(files.filter((_, i) => i !== idx))
  }

  return (
    <div style={{ animation: 'fadeIn 0.4s ease' }}>
      <div className="upload-section">
        {/* Drop Zone */}
        <div
          {...getRootProps()}
          className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="upload-icon">
            <Upload size={22} />
          </div>
          <div className="upload-title">
            Drop medical images here
          </div>
          <div className="upload-subtitle">
            or <span className="upload-browse">browse files</span>
            <br />
            Supports PNG, JPEG, TIFF, DICOM
          </div>
          {files.length > 0 && (
            <div className="file-chips">
              {files.map((f, i) => (
                <div key={i} className="file-chip">
                  <FileImage size={12} />
                  {f.name.length > 25 ? f.name.slice(0, 22) + '...' : f.name}
                  <button onClick={(e) => { e.stopPropagation(); removeFile(i) }}>
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Clinical Input */}
        <div className="card">
          <div className="card-header">
            <h3>
              <FileImage size={16} />
              Clinical Details
            </h3>
          </div>
          <div className="card-body clinical-input">
            <div className="patient-row">
              <div className="input-group">
                <label>Patient Name</label>
                <input
                  type="text"
                  placeholder="Anonymous"
                  value={patientName}
                  onChange={e => onPatientNameChange(e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>Age</label>
                <input
                  type="text"
                  placeholder="e.g. 45"
                  value={patientAge}
                  onChange={e => onPatientAgeChange(e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>Sex</label>
                <select
                  value={patientSex}
                  onChange={e => onPatientSexChange(e.target.value)}
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '10px 14px',
                    color: 'var(--text-primary)',
                    outline: 'none',
                  }}
                >
                  <option value="">--</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <div className="input-group">
              <label>Clinical Notes / Prescription Text</label>
              <textarea
                placeholder="Enter clinical text, symptoms, or prescription details...&#10;e.g. Patient presents with persistent cough, fever 101°F, prescribed Amoxicillin 500mg..."
                value={clinicalText}
                onChange={e => onClinicalTextChange(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 16px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: 'var(--radius)',
          marginBottom: 16,
          color: '#f87171',
          fontSize: '0.85rem',
        }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Analyze Button */}
      <button
        className="btn-analyze"
        onClick={onAnalyze}
        disabled={disabled || (!files.length && !clinicalText.trim())}
      >
        <Send size={18} />
        Analyze & Generate
      </button>
    </div>
  )
}
