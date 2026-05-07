import { Activity } from 'lucide-react'

interface Props {
  online: boolean
  gpuName: string
}

export default function Header({ online, gpuName }: Props) {
  return (
    <header className="app-header">
      <div className="logo-area">
        <div className="logo-icon">
          <Activity size={18} />
        </div>
        <div className="logo-text">
          MedVis<span>-X</span>
        </div>
        <div className="header-badge">
          <span>⚠</span> Research Prototype
        </div>
      </div>

      <div className="header-right">
        {gpuName && (
          <span className="status-text" style={{ fontSize: '0.72rem', color: '#64748b' }}>
            {gpuName}
          </span>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div className={`status-dot`} style={{
            background: online ? '#10b981' : '#ef4444',
            boxShadow: online ? '0 0 6px #10b981' : '0 0 6px #ef4444',
          }} />
          <span className="status-text">
            {online ? 'System Online' : 'Connecting...'}
          </span>
        </div>
      </div>
    </header>
  )
}
