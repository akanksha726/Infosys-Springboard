import { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

// ─── Data Context Builder ────────────────────────────────────────────────────
// Turns the dashboard JSON into a structured text context the bot can search.
function buildContext(data) {
  if (!data) return null;

  const mo = data.market_overview || {};
  const bi = data.brand_insights || {};
  const ci = data.consumer_insights || {};
  const ti = data.topic_insights || {};
  const fc = data.forecast || {};

  // All brand names
  const allBrands = [
    ...(bi.brand_direction || []).map(b => b.brand),
    ...(ci.brand_sentiment || []).map(b => b.brand),
  ];
  const uniqueBrands = [...new Set(allBrands)];

  // Brand data map
  const brandMap = {};
  uniqueBrands.forEach(brand => {
    const dir = bi.brand_direction?.find(b => b.brand === brand);
    const sent = ci.brand_sentiment?.find(b => b.brand === brand);
    const fc_brand = fc.brand_forecast?.brand_forecasts?.find(b => b.brand === brand);
    brandMap[brand] = {
      direction: dir?.direction || 'Unknown',
      sentiment: sent?.final_consumer_sentiment?.toFixed(3) || 'N/A',
      slope: fc_brand?.trend_slope?.toFixed(3) || 'N/A',
      forecast: fc_brand?.trend_direction || 'N/A',
    };
  });

  // Sources / news
  const sources = ci.sources || [];

  return { mo, bi, ci, ti, fc, brandMap, uniqueBrands, sources };
}

// ─── Query Engine ────────────────────────────────────────────────────────────
function answerQuery(query, ctx) {
  if (!ctx) return "Dashboard data hasn't loaded yet. Please wait a moment and try again.";

  const q = query.toLowerCase();
  const { mo, bi, ti, fc, brandMap, uniqueBrands, sources, ci } = ctx;

  // ── helper ──────────────────────────────────────────────────────────────────
  const listBrands = (filter, limit = 8) =>
    uniqueBrands
      .filter(b => filter(brandMap[b]))
      .slice(0, limit)
      .map(b => {
        const d = brandMap[b];
        return `• ${b.toUpperCase()} — Sentiment: ${d.sentiment}, Direction: ${d.direction}, Forecast: ${d.forecast}`;
      })
      .join('\n');

  // ── GREET ──────────────────────────────────────────────────────────────────
  if (['hi', 'hello', 'hey', 'sup', 'namaste'].some(g => q.includes(g))) {
    return `👋 Hi there! I'm your Market Intelligence Assistant.\n\nYou can ask me about:\n• Overall market trend & sentiment\n• Any specific brand (e.g. "tell me about flipkart")\n• Top performing or falling brands\n• Topic trends (e.g. "what's trending in discounts?")\n• Alerts & risks\n• News & sources\n• 7/30/90-day forecasts\n\nWhat would you like to know?`;
  }

  // ── HELP ──────────────────────────────────────────────────────────────────
  if (q.includes('help') || q.includes('what can you') || q.includes('what do you know')) {
    return `I can answer questions about:\n\n📊 **Market Overview** — trend direction, sentiment, volatility\n🏢 **Brands** — any brand's sentiment, direction, forecast\n📈 **Topics** — rising/falling topics like discounts, logistics, funding\n⚠️ **Alerts** — falling/rising momentum brands\n📰 **News** — source articles and headlines\n🔮 **Forecast** — 7-day, 30-day, 90-day predictions\n\nJust ask naturally! e.g. "How is Meesho doing?" or "What brands are rising?"`;
  }

  // ── MARKET OVERVIEW ────────────────────────────────────────────────────────
  if (q.includes('market') || q.includes('overview') || q.includes('overall trend') || q.includes('how is the market')) {
    return `📊 **Current Market Overview**\n\n• Trend Direction: **${mo.trend_direction || 'N/A'}**\n• Sentiment Score: **${mo.current_sentiment?.toFixed(3) || 'N/A'}** (0 = very negative, 1 = very positive)\n• Volatility Index: **${mo.volatility?.toFixed(2) || 'N/A'}**\n• Trend Slope: **${mo.trend_slope > 0 ? '+' : ''}${(mo.trend_slope * 100)?.toFixed(1) || '0'}%**\n\n7-Day Forecast: ${fc['7_day']?.slice(0, 3).map(v => v.toFixed(3)).join(', ') || 'N/A'}...`;
  }

  // ── SENTIMENT SCORE ─────────────────────────────────────────────────────────
  if (q.includes('sentiment')) {
    if (q.includes('brand') || q.includes('all brand') || q.includes('compare')) {
      const top3 = (ci.brand_sentiment || []).slice(0, 5).map(b =>
        `• ${b.brand.toUpperCase()}: ${b.final_consumer_sentiment.toFixed(3)}`
      ).join('\n');
      return `📊 **Brand Consumer Sentiment (Top 5)**\n\n${top3}\n\nHighest: ${bi.top_positive_brand?.toUpperCase() || 'N/A'}\nLowest: ${bi.top_negative_brand?.toUpperCase() || 'N/A'}`;
    }
    return `📊 Overall market sentiment score is **${mo.current_sentiment?.toFixed(3) || 'N/A'}**.\n\nTop positive brand: **${bi.top_positive_brand?.toUpperCase() || 'N/A'}**\nTop negative brand: **${bi.top_negative_brand?.toUpperCase() || 'N/A'}**\nMost volatile: **${bi.most_volatile_brand?.toUpperCase() || 'N/A'}**`;
  }

  // ── VOLATILITY ─────────────────────────────────────────────────────────────
  if (q.includes('volatil')) {
    return `📊 **Market Volatility Index: ${mo.volatility?.toFixed(2) || 'N/A'}**\n\nMost volatile brand: **${bi.most_volatile_brand?.toUpperCase() || 'N/A'}**\n\nA higher volatility index means the market is less stable and more unpredictable.`;
  }

  // ── TOP BRANDS ─────────────────────────────────────────────────────────────
  if (q.includes('top brand') || q.includes('best brand') || q.includes('highest') || q.includes('performing brand')) {
    const list = (bi.top_brands || []).slice(0, 5).map(b =>
      `• ${b.brand.toUpperCase()} — Score: ${b.final_trend_score.toFixed(3)}`
    ).join('\n');
    return `🏆 **Top 5 Performing Brands**\n\n${list}\n\nBest overall: **${bi.top_positive_brand?.toUpperCase() || 'N/A'}**`;
  }

  // ── FALLING BRANDS ─────────────────────────────────────────────────────────
  if (q.includes('falling') || q.includes('declining') || q.includes('worst') || q.includes('negative brand') || q.includes('down')) {
    const falling = listBrands(d => d.direction === 'Falling');
    return falling
      ? `📉 **Falling Brands**\n\n${falling}\n\nWorst performing: **${bi.top_negative_brand?.toUpperCase() || 'N/A'}**`
      : 'No brands are currently showing a falling trend.';
  }

  // ── RISING BRANDS ──────────────────────────────────────────────────────────
  if (q.includes('rising') || q.includes('growing') || q.includes('up brand') || q.includes('gaining')) {
    const rising = listBrands(d => d.direction === 'Rising');
    return rising
      ? `📈 **Rising Brands**\n\n${rising}`
      : 'No brands are currently showing a rising trend.';
  }

  // ── ALL BRANDS LIST ────────────────────────────────────────────────────────
  if (q.includes('all brand') || q.includes('list brand') || q.includes('brands do you') || q.includes('which brand')) {
    const allList = uniqueBrands.map(b => {
      const d = brandMap[b];
      const icon = d.direction === 'Rising' ? '📈' : d.direction === 'Falling' ? '📉' : '➡️';
      return `${icon} ${b.toUpperCase()} (${d.direction})`;
    }).join('\n');
    return `🏢 **All Tracked Brands (${uniqueBrands.length})**\n\n${allList}`;
  }

  // ── FORECAST ───────────────────────────────────────────────────────────────
  if (q.includes('forecast') || q.includes('predict') || q.includes('future') || q.includes('next') || q.includes('7 day') || q.includes('30 day') || q.includes('90 day')) {
    const d7  = fc['7_day']?.map(v => v.toFixed(3)).join(', ') || 'N/A';
    const d30 = fc['30_day']?.slice(0, 5).map(v => v.toFixed(3)).join(', ') || 'N/A';
    return `🔮 **Market Forecast**\n\n📅 **7-Day:** ${d7}\n📅 **30-Day (first 5):** ${d30}...\n\nTrend: Market is expected to remain **${mo.trend_direction || 'stable'}**.`;
  }

  // ── TOPICS ─────────────────────────────────────────────────────────────────
  if (q.includes('topic') || q.includes('trend') || q.includes('category') || q.includes('discount') || q.includes('logistic') || q.includes('funding') || q.includes('technolog') || q.includes('expansion')) {
    return `📌 **Topic Intelligence**\n\n• Top Topics: **${(ti.top_topics || []).join(', ') || 'N/A'}**\n• Fastest Rising: **${ti.fastest_rising_topic || 'N/A'}**\n• Fastest Declining: **${ti.fastest_declining_topic || 'N/A'}**\n\nThe **${ti.fastest_rising_topic || 'N/A'}** category is gaining the most traction right now.`;
  }

  // ── ALERTS / RISKS ─────────────────────────────────────────────────────────
  if (q.includes('alert') || q.includes('warning') || q.includes('risk') || q.includes('signal') || q.includes('danger')) {
    const falling = (bi.brand_direction || []).filter(b => b.direction === 'Falling').map(b => b.brand.toUpperCase());
    const risks = data => data?.risk_signals || '';
    return `⚠️ **Active Alerts**\n\n${falling.length > 0 ? `Falling Brands: **${falling.slice(0, 5).join(', ')}**` : 'No falling brands.'}\n\nTop Negative Brand: **${bi.top_negative_brand?.toUpperCase() || 'N/A'}**\nMost Volatile: **${bi.most_volatile_brand?.toUpperCase() || 'N/A'}**\n\nFor detailed risk signals, check the Alerts page.`;
  }

  // ── NEWS / SOURCES ─────────────────────────────────────────────────────────
  if (q.includes('news') || q.includes('article') || q.includes('headline') || q.includes('source')) {
    if (sources.length === 0) return 'No news sources are currently loaded. Run the market engine to fetch fresh news.';
    const snippets = sources.slice(0, 3).map((s, i) => {
      const lines = (s.content || '').split('\n').filter(Boolean);
      const headline = lines.find(l => l.toLowerCase().startsWith('news:'))?.replace(/^news:\s*/i, '') || s.content?.slice(0, 100);
      return `${i + 1}. ${headline?.trim() || 'No headline'}`;
    }).join('\n');
    return `📰 **Recent News Headlines (${sources.length} total)**\n\n${snippets}\n\nVisit the **News** or **Research Sources** pages for all articles with links.`;
  }

  // ── AI INSIGHTS ────────────────────────────────────────────────────────────
  if (q.includes('insight') || q.includes('ai insight') || q.includes('summary') || q.includes('what does ai say') || q.includes('ai opinion')) {
    const aiInsight = (typeof ci.consumer_ai_insight === 'string') ? ci.consumer_ai_insight : null;
    if (aiInsight) return `🤖 **AI Market Insight**\n\n${aiInsight.slice(0, 600)}...`;
    return 'AI insights are available in the Overview page under the AI Market Insight section.';
  }

  // ─── SPECIFIC BRAND LOOKUP ─────────────────────────────────────────────────
  const foundBrand = uniqueBrands.find(b => q.includes(b.toLowerCase()));
  if (foundBrand) {
    const d = brandMap[foundBrand];
    const icon = d.direction === 'Rising' ? '📈' : d.direction === 'Falling' ? '📉' : '➡️';

    // Related news from sources
    const relatedNews = sources
      .filter(s => (s.content || '').toLowerCase().includes(foundBrand.toLowerCase()))
      .slice(0, 2)
      .map(s => {
        const lines = (s.content || '').split('\n').filter(Boolean);
        const headline = lines.find(l => l.toLowerCase().startsWith('news:'))?.replace(/^news:\s*/i, '') || '';
        const sentiment = lines.find(l => l.toLowerCase().startsWith('sentiment:'))?.replace(/^sentiment:\s*/i, '') || '';
        return `  • ${headline.trim()} (${sentiment})`;
      }).join('\n');

    return `${icon} **${foundBrand.toUpperCase()}**\n\n• Trend Direction: **${d.direction}**\n• Consumer Sentiment: **${d.sentiment}** / 1.0\n• Forecast Trend: **${d.forecast}**\n• Forecast Slope: **${d.slope}**\n\n${relatedNews ? `📰 **Related News:**\n${relatedNews}` : 'No specific news available for this brand right now.'}\n\nFor the full brand comparison, visit the **Brand Comparison** page.`;
  }

  // ─── FALLBACK ──────────────────────────────────────────────────────────────
  // Try to find any source that matches keywords
  const keywords = q.split(/\s+/).filter(w => w.length > 3);
  const matchingSources = sources.filter(s =>
    keywords.some(kw => (s.content || '').toLowerCase().includes(kw))
  );
  if (matchingSources.length > 0) {
    const s = matchingSources[0];
    const lines = (s.content || '').split('\n').filter(Boolean);
    const headline = lines.find(l => l.toLowerCase().startsWith('news:'))?.replace(/^news:\s*/i, '') || s.content?.slice(0, 150);
    return `📰 **Found in Market Data:**\n\n${headline?.trim() || 'No headline'}\n\n${s.url ? `🔗 Read more at: ${s.url}` : ''}`;
  }

  return `I couldn't find specific data for "${query}". Try asking:\n• "How is [brand name] doing?"\n• "What is the market trend?"\n• "Which brands are falling?"\n• "What topics are trending?"\n• "Show me the forecast"\n• "List all brands"`;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function Chatbot({ data }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { text: "👋 Hi! I'm your Market Intelligence Assistant. Ask me anything about brands, sentiment, topics, forecasts, or news!", isBot: true }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const ctx = buildContext(data);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage = inputValue.trim();
    setMessages(prev => [...prev, { text: userMessage, isBot: false }]);
    setInputValue('');
    setIsTyping(true);

    setTimeout(() => {
      const response = answerQuery(userMessage, ctx);
      setIsTyping(false);
      setMessages(prev => [...prev, { text: response, isBot: true }]);
    }, 500);
  };

  const QUICK_PROMPTS = ['Market overview', 'Top brands', 'Falling brands', 'Topic trends', 'Show forecast'];

  return (
    <>
      <button
        className={`chatbot-toggle-btn ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {isOpen ? (
            <path d="M18 6L6 18M6 6l12 12" />
          ) : (
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          )}
        </svg>
      </button>

      {isOpen && (
        <div className="chatbot-window glass-panel animated-enter">
          <div className="chatbot-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: data ? '#10b981' : '#f59e0b', boxShadow: data ? '0 0 6px #10b981' : '0 0 6px #f59e0b' }} />
              <h3 style={{ margin: 0 }}>Market Assistant</h3>
            </div>
            <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{data ? 'Connected to live data' : 'Data loading...'}</span>
          </div>

          <div className="chatbot-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chatbot-message ${msg.isBot ? 'bot' : 'user'}`} style={{ whiteSpace: 'pre-wrap' }}>
                {msg.text}
              </div>
            ))}
            {isTyping && (
              <div className="chatbot-message bot" style={{ display: 'flex', gap: '4px', padding: '0.75rem 1rem' }}>
                <span style={{ animation: 'pulse 1s infinite', animationDelay: '0s' }}>●</span>
                <span style={{ animation: 'pulse 1s infinite', animationDelay: '0.2s' }}>●</span>
                <span style={{ animation: 'pulse 1s infinite', animationDelay: '0.4s' }}>●</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick prompts */}
          {messages.length <= 2 && (
            <div style={{ padding: '0.5rem 0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              {QUICK_PROMPTS.map(p => (
                <button
                  key={p}
                  onClick={() => { setInputValue(p); }}
                  style={{ fontSize: '0.7rem', padding: '0.2rem 0.6rem', borderRadius: '999px', background: 'rgba(99,102,241,0.15)', color: '#a78bfa', border: '1px solid rgba(99,102,241,0.3)', cursor: 'pointer', transition: 'all 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.3)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'rgba(99,102,241,0.15)'}
                >
                  {p}
                </button>
              ))}
            </div>
          )}

          <form className="chatbot-input-area" onSubmit={handleSend}>
            <input
              type="text"
              className="glass-select chatbot-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about brands, trends, news..."
            />
            <button type="submit" className="glass-btn primary chatbot-send-btn" disabled={isTyping}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </form>
          <style>{`@keyframes pulse { 0%,100% { opacity:0.3 } 50% { opacity:1 } }`}</style>
        </div>
      )}
    </>
  );
}
