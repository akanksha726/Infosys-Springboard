import { useState } from 'react';
import { Newspaper, ExternalLink, Search, Filter } from 'lucide-react';

const SENTIMENT_COLORS = {
  positive: { bg: 'rgba(16, 185, 129, 0.1)', border: '#10b981', dot: '#10b981', text: '#10b981' },
  negative: { bg: 'rgba(239, 68, 68, 0.1)',  border: '#ef4444', dot: '#ef4444', text: '#ef4444' },
  neutral:  { bg: 'rgba(245, 158, 11, 0.1)',  border: '#f59e0b', dot: '#f59e0b', text: '#f59e0b' },
};

function parseSources(data) {
  // Collect all sources from every section of the data
  const allSources = [];
  const seen = new Set();

  const addSources = (sources) => {
    if (!Array.isArray(sources)) return;
    sources.forEach(s => {
      const key = s.url || s.content;
      if (key && seen.has(key)) return;
      if (key) seen.add(key);

      // Parse the content string: "\nBrand: flipkart\nTopic: funding\nSentiment: neutral\nNews: ..."
      const raw = s.content || '';
      const lines = raw.split('\n').filter(Boolean);
      const fields = {};
      lines.forEach(line => {
        const colon = line.indexOf(':');
        if (colon > -1) {
          const k = line.substring(0, colon).trim().toLowerCase();
          const v = line.substring(colon + 1).trim();
          fields[k] = v;
        }
      });

      allSources.push({
        source_id: s.source_id,
        brand: fields['brand'] || 'Unknown',
        topic: fields['topic'] || 'general',
        sentiment: (fields['sentiment'] || 'neutral').toLowerCase(),
        headline: fields['news'] || raw.slice(0, 120),
        url: s.url || '#',
      });
    });
  };

  addSources(data?.consumer_insights?.sources);

  // Also pull from any other sources arrays if they exist
  const allInsightSources = data?.consumer_insights?.consumer_ai_insight;
  if (Array.isArray(allInsightSources) && Array.isArray(allInsightSources[1])) {
    addSources(allInsightSources[1]);
  }

  return allSources;
}

export default function News({ data }) {
  const [search, setSearch] = useState('');
  const [filterSentiment, setFilterSentiment] = useState('all');
  const [filterBrand, setFilterBrand] = useState('all');

  if (!data) return <div className="p-8 animated-enter text-center">Loading News...</div>;

  const allNews = parseSources(data);
  const brands = ['all', ...new Set(allNews.map(n => n.brand))].filter(Boolean);

  const filtered = allNews.filter(n => {
    const matchSearch = n.headline.toLowerCase().includes(search.toLowerCase()) ||
                        n.brand.toLowerCase().includes(search.toLowerCase()) ||
                        n.topic.toLowerCase().includes(search.toLowerCase());
    const matchSentiment = filterSentiment === 'all' || n.sentiment === filterSentiment;
    const matchBrand = filterBrand === 'all' || n.brand === filterBrand;
    return matchSearch && matchSentiment && matchBrand;
  });

  return (
    <div className="animated-enter">
      <div className="page-header">
        <div>
          <h1 className="page-title">News & Market Intelligence</h1>
          <p className="text-secondary mt-1">
            Live news articles analysed by the AI pipeline — {allNews.length} sources indexed.
          </p>
        </div>
        <div className="page-actions" style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search */}
          <div style={{ position: 'relative' }}>
            <Search size={15} style={{ position: 'absolute', top: '50%', left: '10px', transform: 'translateY(-50%)', color: '#94a3b8' }} />
            <input
              type="text"
              placeholder="Search news..."
              className="glass-btn"
              style={{ paddingLeft: '2rem', textAlign: 'left', width: '200px' }}
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          {/* Sentiment Filter */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Filter size={14} style={{ color: '#94a3b8' }} />
            <select
              className="glass-select"
              value={filterSentiment}
              onChange={e => setFilterSentiment(e.target.value)}
            >
              <option value="all">All Sentiment</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
            </select>
          </div>

          {/* Brand Filter */}
          <select
            className="glass-select"
            value={filterBrand}
            onChange={e => setFilterBrand(e.target.value)}
          >
            {brands.map(b => (
              <option key={b} value={b}>{b === 'all' ? 'All Brands' : b.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Count badge */}
      <div style={{ marginBottom: '1.5rem', color: '#94a3b8', fontSize: '0.85rem' }}>
        Showing <strong style={{ color: '#e2e8f0' }}>{filtered.length}</strong> of {allNews.length} articles
      </div>

      {/* News Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1.25rem' }}>
        {filtered.map((article, idx) => {
          const colors = SENTIMENT_COLORS[article.sentiment] || SENTIMENT_COLORS.neutral;
          return (
            <div
              key={idx}
              className="glass-panel"
              style={{
                padding: '1.25rem 1.5rem',
                borderLeft: `4px solid ${colors.border}`,
                background: colors.bg,
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >
              {/* Header Row */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  <span style={{
                    background: 'rgba(59,130,246,0.15)', color: '#60a5fa',
                    padding: '0.2rem 0.6rem', borderRadius: '999px', fontSize: '0.7rem',
                    fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em'
                  }}>
                    {article.brand}
                  </span>
                  <span style={{
                    background: 'rgba(139,92,246,0.15)', color: '#a78bfa',
                    padding: '0.2rem 0.6rem', borderRadius: '999px', fontSize: '0.7rem',
                    fontWeight: 600, textTransform: 'capitalize'
                  }}>
                    {article.topic}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <span style={{
                    width: '8px', height: '8px', borderRadius: '50%',
                    background: colors.dot, boxShadow: `0 0 6px ${colors.dot}`
                  }} />
                  <span style={{ fontSize: '0.75rem', color: colors.text, fontWeight: 600, textTransform: 'capitalize' }}>
                    {article.sentiment}
                  </span>
                </div>
              </div>

              {/* Headline */}
              <p style={{ fontSize: '0.875rem', color: '#e2e8f0', lineHeight: '1.6', flex: 1 }}>
                {article.headline}
              </p>

              {/* Link */}
              {article.url && article.url !== '#' && (
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
                    fontSize: '0.775rem', color: '#60a5fa', fontWeight: 600,
                    textDecoration: 'none', transition: 'color 0.15s'
                  }}
                  onMouseEnter={e => e.currentTarget.style.color = '#93c5fd'}
                  onMouseLeave={e => e.currentTarget.style.color = '#60a5fa'}
                >
                  Read Full Article <ExternalLink size={12} />
                </a>
              )}
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div style={{ textAlign: 'center', padding: '4rem 2rem', color: '#94a3b8' }}>
          <Newspaper size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
          <p style={{ fontSize: '1.1rem' }}>No news articles match your filters.</p>
          <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>Try adjusting your search or run the market engine to refresh data.</p>
        </div>
      )}
    </div>
  );
}
