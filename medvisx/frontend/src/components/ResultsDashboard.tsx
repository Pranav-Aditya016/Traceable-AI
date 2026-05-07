import { useState } from 'react'
import {
  BarChart3, Image, Brain, FileText, Clock, Copy,
  Download, RotateCcw, Shield, Stethoscope, Pill, AlertTriangle,
  CheckCircle, TrendingUp, Eye, Activity, History, ThumbsUp, ThumbsDown
} from 'lucide-react'
import type { AnalysisResult } from '../types'

interface Props {
  result: AnalysisResult
  onReset: () => void
}

type ExplanationTone = 'concise' | 'technical' | 'patient'
type ImageView = 'generated' | 'heatmap' | 'shap'

export default function ResultsDashboard({ result, onReset }: Props) {
  const [tone, setTone] = useState<ExplanationTone>('concise')
  const [imageView, setImageView] = useState<ImageView>('generated')
  const [copied, setCopied] = useState(false)

  const { prediction, xai, images, entities, ocr, timings, patient, provenance, auditLog } = result

  const confidenceLevel = prediction.confidence >= 0.7 ? 'high'
    : prediction.confidence >= 0.4 ? 'medium' : 'low'

  const copyExplanation = () => {
    navigator.clipboard.writeText(xai.explanation)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadImage = (b64: string, name: string) => {
    const a = document.createElement('a')
    a.href = `data:image/png;base64,${b64}`
    a.download = name
    a.click()
  }

  // Generate different tones from the full explanation
  const getExplanation = (t: ExplanationTone): string => {
    const full = xai.explanation
    const summary = xai.naturalLanguageSummary || ''
    if (t === 'concise') {
      // Use the AI-generated natural language summary if available
      if (summary) return summary
      const paras = full.split('\n\n').filter(Boolean)
      return paras.slice(0, 3).join('\n\n')
    }
    if (t === 'technical') {
      return full
    }
    // patient-friendly
    return `Based on the analysis, the most likely finding is ${prediction.top_disease} ` +
      `with a confidence level of ${(prediction.confidence * 100).toFixed(0)}%.\n\n` +
      `The AI examined the medical document you provided and identified key clinical indicators ` +
      `related to the ${prediction.organ || 'affected area'}. ` +
      `A synthetic medical image was generated for visualization purposes only.\n\n` +
      `Important: This is a research tool and should not replace professional medical diagnosis. ` +
      `Please consult your healthcare provider for clinical decisions.`
  }

  return (
    <div style={{ animation: 'fadeIn 0.5s ease' }}>
      {/* Top bar with actions */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Analysis Results</h2>
          <span style={{
            padding: '3px 10px',
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
            borderRadius: 20,
            fontSize: '0.72rem',
            fontWeight: 600,
            color: '#10b981',
          }}>
            <CheckCircle size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
            Complete
          </span>
          <span style={{
            fontSize: '0.75rem',
            color: 'var(--text-dim)',
          }}>
            {result.created_at} · Report #{result.report_id.slice(4, 12)}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn-icon" onClick={onReset} title="New Analysis">
            <RotateCcw size={15} />
          </button>
        </div>
      </div>

      {/* ── Row 1: Summary + Prediction ────────────────────────────── */}
      <div className="results-grid" style={{ marginBottom: 20 }}>
        {/* Analysis Summary Card */}
        <div className="card">
          <div className="card-header">
            <h3><BarChart3 size={16} /> Analysis Summary</h3>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              {patient.name}{patient.age ? `, ${patient.age}` : ''}{patient.sex ? ` ${patient.sex}` : ''}
            </span>
          </div>
          <div className="card-body">
            <div className="summary-grid">
              <div className="summary-item">
                <div className="label">Primary Finding</div>
                <div className="value accent">{prediction.top_disease}</div>
              </div>
              <div className="summary-item">
                <div className="label">Confidence</div>
                <div className={`value ${confidenceLevel === 'high' ? 'green' : confidenceLevel === 'medium' ? 'amber' : ''}`}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-meter">
                  <div
                    className={`confidence-fill ${confidenceLevel}`}
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
              </div>
              <div className="summary-item">
                <div className="label">Imaging Modality</div>
                <div className="value cyan">{prediction.modality}</div>
              </div>
              {prediction.organ && (
                <div className="summary-item">
                  <div className="label">Target Organ</div>
                  <div className="value cyan">{prediction.organ}</div>
                </div>
              )}
              <div className="summary-item">
                <div className="label">Analysis Engine</div>
                <div className="value" style={{ fontSize: '0.9rem' }}>
                  {ocr.method_used === 'llava_multimodal' ? 'LLaVA + Mistral' :
                   ocr.method_used === 'easyocr' ? 'EasyOCR' :
                   ocr.method_used === 'trained_crnn' ? 'Trained CRNN' : 'Manual Input'}
                </div>
              </div>
            </div>

            {/* AI Summary */}
            {xai.naturalLanguageSummary && (
              <div style={{
                marginTop: 14, padding: '10px 14px',
                background: 'rgba(59, 130, 246, 0.06)',
                border: '1px solid rgba(59, 130, 246, 0.15)',
                borderRadius: 8, fontSize: '0.82rem',
                lineHeight: 1.6, color: 'var(--text-primary)',
              }}>
                <div style={{
                  fontSize: '0.68rem', fontWeight: 600, color: 'var(--accent)',
                  textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6,
                }}>
                  <Activity size={11} style={{ marginRight: 4, verticalAlign: -2 }} />
                  AI Summary
                </div>
                {xai.naturalLanguageSummary}
              </div>
            )}

            {/* Hypotheses */}
            {prediction.hypotheses.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{
                  fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8,
                }}>
                  Differential Diagnoses
                </div>
                <div className="hypothesis-list">
                  {prediction.hypotheses.slice(0, 4).map((h, i) => (
                    <div key={i} className="hypothesis-item">
                      <span className="name">
                        <TrendingUp size={13} style={{ marginRight: 6, color: 'var(--accent)', verticalAlign: -2 }} />
                        {h.disease}
                        {h.model && (
                          <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginLeft: 6 }}>
                            ({h.model})
                          </span>
                        )}
                      </span>
                      <span className="score">{(h.score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Extracted Data Card */}
        <div className="card">
          <div className="card-header">
            <h3><FileText size={16} /> Extracted Data</h3>
          </div>
          <div className="card-body extracted-data">
            {/* OCR Text */}
            <div className="extracted-section">
              <h4>Clinical Text</h4>
              <div className="ocr-text-preview">{ocr.final_text}</div>
            </div>

            {/* Entities */}
            {entities.symptoms.length > 0 && (
              <div className="extracted-section">
                <h4>
                  <Stethoscope size={12} style={{ marginRight: 4, verticalAlign: -2 }} />
                  Symptoms
                </h4>
                <div className="entity-tags">
                  {entities.symptoms.map((s, i) => (
                    <span key={i} className="entity-tag symptom">
                      <AlertTriangle size={10} /> {s.text}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {entities.medications.length > 0 && (
              <div className="extracted-section">
                <h4>
                  <Pill size={12} style={{ marginRight: 4, verticalAlign: -2 }} />
                  Medications
                </h4>
                <div className="entity-tags">
                  {entities.medications.map((m, i) => (
                    <span key={i} className="entity-tag medication">
                      <Pill size={10} /> {m.text}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {entities.conditions.length > 0 && (
              <div className="extracted-section">
                <h4>
                  <Shield size={12} style={{ marginRight: 4, verticalAlign: -2 }} />
                  Conditions
                </h4>
                <div className="entity-tags">
                  {entities.conditions.map((c, i) => (
                    <span key={i} className="entity-tag condition">{c.text}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Text Features from LLaVA analysis */}
            {xai.text_features && xai.text_features.length > 0 && (
              <div className="extracted-section">
                <h4>
                  <TrendingUp size={12} style={{ marginRight: 4, verticalAlign: -2 }} />
                  Key Clinical Features
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {xai.text_features.map((tf, i) => {
                    const maxW = Math.max(...(xai.text_features?.map(t => Math.abs(t.weight)) || [1]))
                    const barWidth = maxW > 0 ? (Math.abs(tf.weight) / maxW) * 100 : 0
                    return (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ flex: '0 0 140px', fontSize: '0.78rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                          {tf.feature}
                        </span>
                        <div style={{
                          flex: 1, height: 6, borderRadius: 3,
                          background: 'var(--bg-tertiary)',
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${barWidth}%`, height: '100%', borderRadius: 3,
                            background: tf.weight >= 0 ? 'var(--accent)' : 'var(--red, #ef4444)',
                          }} />
                        </div>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', minWidth: 40, textAlign: 'right' }}>
                          {tf.weight >= 0 ? '+' : ''}{tf.weight.toFixed(2)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Row 2: Images + SHAP ───────────────────────────────────── */}
      <div className="results-grid" style={{ marginBottom: 20 }}>
        {/* Localization & Provenance */}
        <div className="card">
          <div className="card-header">
            <h3><Eye size={16} /> Localization & Provenance</h3>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${imageView === 'generated' ? 'active' : ''}`}
                onClick={() => setImageView('generated')}
              >
                Generated
              </button>
              <button
                className={`toggle-btn ${imageView === 'heatmap' ? 'active' : ''}`}
                onClick={() => setImageView('heatmap')}
              >
                Saliency
              </button>
              <button
                className={`toggle-btn ${imageView === 'shap' ? 'active' : ''}`}
                onClick={() => setImageView('shap')}
              >
                SHAP Plot
              </button>
            </div>
          </div>
          <div className="card-body">
            <div className="image-container">
              {imageView === 'generated' && (
                <>
                  <img src={`data:image/png;base64,${images.generated}`} alt="Generated medical image" />
                  <span className="image-overlay-badge synthetic">
                    Synthetic — Not For Diagnosis
                  </span>
                </>
              )}
              {imageView === 'heatmap' && (
                <>
                  <img src={`data:image/png;base64,${images.heatmap}`} alt="Grad-CAM heatmap" />
                  <span className="image-overlay-badge heatmap">
                    Grad-CAM Saliency
                  </span>
                </>
              )}
              {imageView === 'shap' && (
                <img src={`data:image/png;base64,${images.shap_plot}`} alt="SHAP values plot" />
              )}
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
              <button
                className="btn-icon"
                onClick={() => {
                  const img = imageView === 'heatmap' ? images.heatmap
                    : imageView === 'shap' ? images.shap_plot : images.generated
                  downloadImage(img, `medvisx_${imageView}.png`)
                }}
                title="Download Image"
              >
                <Download size={14} />
              </button>
            </div>
            {/* Localization bbox info */}
            {prediction.localization?.bbox && (
              <div style={{
                marginTop: 10, padding: '8px 12px',
                background: 'var(--bg-tertiary)',
                borderRadius: 6, fontSize: '0.75rem',
                color: 'var(--text-muted)',
              }}>
                <strong>Localization bbox:</strong> [{prediction.localization.bbox.join(', ')}]
                {provenance && (
                  <span style={{ marginLeft: 12 }}>
                    <strong>Generator:</strong> {provenance.generator} | <strong>Seed:</strong> {provenance.seed}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* SHAP Values Table */}
        <div className="card">
          <div className="card-header">
            <h3><BarChart3 size={16} /> Feature Attribution (SHAP)</h3>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="shap-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody>
                {xai.shap_values.map((sv, i) => {
                  const maxImpact = Math.max(...xai.shap_values.map(v => Math.abs(v.impact)))
                  const barWidth = maxImpact > 0 ? (Math.abs(sv.impact) / maxImpact) * 100 : 0
                  const isPos = sv.impact >= 0

                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                        {sv.feature}
                      </td>
                      <td>
                        <div className="impact-bar">
                          <div
                            className={`impact-fill ${isPos ? 'positive' : 'negative'}`}
                            style={{ width: `${barWidth}%`, minWidth: barWidth > 0 ? 4 : 0 }}
                          />
                          <span className={`impact-value ${isPos ? 'positive' : 'negative'}`}>
                            {isPos ? '+' : ''}{sv.impact.toFixed(3)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── Row 3: Explanation ──────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <h3><Brain size={16} /> Natural Language Explanation</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${tone === 'concise' ? 'active' : ''}`}
                onClick={() => setTone('concise')}
              >
                Concise
              </button>
              <button
                className={`toggle-btn ${tone === 'technical' ? 'active' : ''}`}
                onClick={() => setTone('technical')}
              >
                Technical
              </button>
              <button
                className={`toggle-btn ${tone === 'patient' ? 'active' : ''}`}
                onClick={() => setTone('patient')}
              >
                Patient-Friendly
              </button>
            </div>
            <button className="btn-icon" onClick={copyExplanation} title="Copy">
              {copied ? <CheckCircle size={14} style={{ color: 'var(--green)' }} /> : <Copy size={14} />}
            </button>
          </div>
        </div>
        <div className="card-body">
          <div className="explanation-box">
            {getExplanation(tone)}
          </div>
        </div>
      </div>

      {/* ── Audit Log ──────────────────────────────────────────────── */}
      {auditLog && auditLog.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <h3><History size={16} /> Audit Log</h3>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="shap-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Action</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {auditLog.map((entry, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{entry.user}</td>
                    <td>{entry.action}</td>
                    <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {new Date(entry.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Timing Footer ──────────────────────────────────────────── */}
      <div className="timing-bar">
        <Clock size={14} style={{ color: 'var(--text-muted)' }} />
        {Object.entries(timings).map(([key, val]) => (
          <div key={key} className="timing-item">
            <span className="timing-label">{key.charAt(0).toUpperCase() + key.slice(1)}:</span>
            <span className="timing-value">{val}s</span>
          </div>
        ))}
      </div>

      {/* Disclaimer */}
      <div style={{
        textAlign: 'center',
        padding: '20px 0 10px',
        fontSize: '0.72rem',
        color: 'var(--text-dim)',
      }}>
        <Shield size={12} style={{ verticalAlign: -2, marginRight: 4 }} />
        This is a research prototype. Synthetic images are AI-generated and must not be used for clinical diagnosis.
      </div>
    </div>
  )
}
