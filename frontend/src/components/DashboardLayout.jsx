import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import { RefreshCw, AlertCircle } from 'lucide-react';

export default function DashboardLayout({ lastUpdated, isError, isSyncing, onRefresh }) {
  const formatTime = (date) => {
    if (!date) return 'Just now';
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        {isError && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5', padding: '0.75rem 1rem', borderRadius: '8px', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={18} />
            <span>Connection to AI pipeline lost. Retrying automatically...</span>
          </div>
        )}

        <header style={{ display: 'flex', justifyContent: 'flex-end', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.03)', padding: '0.4rem 0.8rem', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>
            {isSyncing ? (
              <RefreshCw size={14} style={{ color: '#94a3b8', animation: 'spin 1s linear infinite' }} />
            ) : (
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: isError ? '#ef4444' : '#10b981', boxShadow: isError ? '0 0 8px #ef4444' : '0 0 8px #10b981' }} />
            )}
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              {isError ? 'Disconnected' : `Live • Updated: ${formatTime(lastUpdated)}`}
            </span>
            <button
              onClick={onRefresh}
              disabled={isSyncing}
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '4px', marginLeft: '4px', display: 'flex', alignItems: 'center', color: '#94a3b8' }}
              title="Refresh Dashboard"
            >
              <RefreshCw size={14} style={{ opacity: 0.7 }} />
            </button>
          </div>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </header>

        <Outlet />
      </main>
    </div>
  );
}
