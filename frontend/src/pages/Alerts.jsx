import { AlertTriangle, TrendingDown, TrendingUp, Zap, Bell } from 'lucide-react';

function SeverityBadge({ severity }) {
  const map = {
    HIGH:   { color: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
    MEDIUM: { color: '#f59e0b', bg: 'rgba(245,158,11,0.15)' },
    LOW:    { color: '#10b981', bg: 'rgba(16,185,129,0.15)' },
    INFO:   { color: '#3b82f6', bg: 'rgba(59,130,246,0.15)' },
  };
  const s = map[severity?.toUpperCase()] || map.INFO;
  return (
    <span style={{
      background: s.bg, color: s.color,
      padding: '0.2rem 0.6rem', borderRadius: '999px',
      fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase'
    }}>
      {severity || 'INFO'}
    </span>
  );
}

export default function Alerts({ data }) {
  if (!data) return <div className="p-8 animated-enter">Loading...</div>;

  // Combine structured alerts + brand direction signals
  const structuredAlerts = data.alerts || [];

  const fallingBrands = data.brand_insights?.brand_direction?.filter(b => b.direction === 'Falling') || [];
  const risingBrands  = data.brand_insights?.brand_direction?.filter(b => b.direction === 'Rising')  || [];

  const overallDirection = data.market_overview?.trend_direction || 'Stable';
  const sentiment        = data.market_overview?.current_sentiment || 0;
  const volatility       = data.market_overview?.volatility || 0;
  const risingTopic      = data.topic_insights?.fastest_rising_topic;
  const decliningTopic   = data.topic_insights?.fastest_declining_topic;

  return (
    <div className="animated-enter">
      <div className="page-header">
        <div>
          <h1 className="page-title">Market Alerts</h1>
          <p className="text-secondary mt-1">AI-generated signals, brand momentum alerts, and market risk indicators.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: 'rgba(255,255,255,0.04)', borderRadius: '999px', border: '1px solid rgba(255,255,255,0.07)' }}>
          <Bell size={14} style={{ color: '#f59e0b' }} />
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            {structuredAlerts.length + fallingBrands.length + risingBrands.length} active signals
          </span>
        </div>
      </div>

      {/* ── MARKET OVERVIEW SIGNAL ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', borderLeft: `4px solid ${overallDirection === 'Bearish' ? '#ef4444' : '#10b981'}`, background: overallDirection === 'Bearish' ? 'rgba(239,68,68,0.05)' : 'rgba(16,185,129,0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Overall Market</span>
            <SeverityBadge severity={overallDirection === 'Bearish' ? 'HIGH' : 'LOW'} />
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.25rem' }}>
            Market is {overallDirection}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Sentiment: {sentiment.toFixed(3)} · Volatility Index: {volatility.toFixed(2)}
          </div>
        </div>

        {risingTopic && (
          <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', borderLeft: '4px solid #10b981', background: 'rgba(16,185,129,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Topic Breakout</span>
              <SeverityBadge severity="MEDIUM" />
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#10b981', marginBottom: '0.25rem', textTransform: 'capitalize' }}>
              ↑ {risingTopic}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Fastest rising topic this cycle</div>
          </div>
        )}

        {decliningTopic && (
          <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', borderLeft: '4px solid #ef4444', background: 'rgba(239,68,68,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Topic Decline</span>
              <SeverityBadge severity="MEDIUM" />
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ef4444', marginBottom: '0.25rem', textTransform: 'capitalize' }}>
              ↓ {decliningTopic}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Fastest declining topic this cycle</div>
          </div>
        )}
      </div>

      {/* ── FALLING BRAND ALERTS ── */}
      {fallingBrands.length > 0 && (
        <section style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, color: '#ef4444', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <TrendingDown size={18} /> Falling Brand Alerts ({fallingBrands.length})
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0.9rem' }}>
            {fallingBrands.map((b, i) => (
              <div key={i} className="glass-panel" style={{ padding: '1rem 1.25rem', borderLeft: '4px solid #ef4444', background: 'rgba(239,68,68,0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 700, color: '#e2e8f0', textTransform: 'capitalize', fontSize: '0.95rem' }}>{b.brand}</span>
                  <SeverityBadge severity="HIGH" />
                </div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Momentum Dropping</div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>FinBERT detects negative sentiment vectors.</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── RISING BRAND SIGNALS ── */}
      {risingBrands.length > 0 && (
        <section style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, color: '#10b981', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <TrendingUp size={18} /> Rising Brand Signals ({risingBrands.length})
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0.9rem' }}>
            {risingBrands.map((b, i) => (
              <div key={i} className="glass-panel" style={{ padding: '1rem 1.25rem', borderLeft: '4px solid #10b981', background: 'rgba(16,185,129,0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 700, color: '#e2e8f0', textTransform: 'capitalize', fontSize: '0.95rem' }}>{b.brand}</span>
                  <SeverityBadge severity="LOW" />
                </div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Positive Momentum</div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>FinBERT detects upward sentiment trajectory.</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── STRUCTURED ALERTS FROM ENGINE ── */}
      {structuredAlerts.length > 0 && (
        <section>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, color: '#f59e0b', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={18} /> Engine Alerts ({structuredAlerts.length})
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {structuredAlerts.map((alert, i) => (
              <div key={i} className="glass-panel" style={{ padding: '1rem 1.25rem', borderLeft: `4px solid #f59e0b`, background: 'rgba(245,158,11,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <span style={{ fontWeight: 700, color: '#e2e8f0', textTransform: 'capitalize', marginRight: '0.5rem' }}>{alert.brand || 'MARKET'}</span>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{alert.message || alert.type}</span>
                </div>
                <SeverityBadge severity={alert.severity} />
              </div>
            ))}
          </div>
        </section>
      )}

      {structuredAlerts.length === 0 && fallingBrands.length === 0 && risingBrands.length === 0 && (
        <div style={{ textAlign: 'center', padding: '4rem', color: '#94a3b8' }}>
          <Zap size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
          <p>No active market alerts at this time.</p>
        </div>
      )}
    </div>
  );
}
