import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, MessageSquare, Activity, AlertCircle, FileText, Download } from 'lucide-react';
import AnimatedNumber from '../components/AnimatedNumber';

export default function Overview({ data }) {
  const [selectedBrand, setSelectedBrand] = useState('overall');
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (!data) return;

    // Based on whether we look at overall forecast or brand
    if (selectedBrand === 'overall') {
      const forecast = data.forecast?.['30_day'] || [];
      const formatted = forecast.map((val, idx) => ({
        day: `Day ${idx + 1}`,
        sentiment: val,
      }));
      setChartData(formatted);
    } else {
      const brandForecast = data.forecast?.brand_forecast?.brand_forecasts?.find(f => f.brand === selectedBrand);
      if (brandForecast) {
        const forecast = brandForecast.forecasts['30_day'] || [];
        const formatted = forecast.map((val, idx) => ({
          day: `Day ${idx + 1}`,
          sentiment: val,
        }));
        setChartData(formatted);
      } else {
        setChartData([]);
      }
    }
  }, [data, selectedBrand]);

  if (!data) return <div className="p-8 animated-enter text-center">Loading Market Data...</div>;

  const handleExport = () => {
    const { market_overview, brand_insights, consumer_insights, forecast } = data;
    
    const rows = [
      ['AI Ecommerce Trend Forecaster Report'],
      [''],
      ['Market Overview'],
      ['Trend Direction', market_overview?.trend_direction || 'N/A'],
      ['Trend Slope', market_overview?.trend_slope || '0'],
      ['Current Sentiment', market_overview?.current_sentiment || '0'],
      ['Volatility', market_overview?.volatility || '0'],
      [''],
      ['Brand Performance'],
      ['Brand', 'Consumer Sentiment', 'Direction', 'Forecast Slope']
    ];

    const allBrandsSet = new Set();
    brand_insights?.brand_direction?.forEach(b => allBrandsSet.add(b.brand));
    consumer_insights?.brand_sentiment?.forEach(b => allBrandsSet.add(b.brand));
    forecast?.brand_forecast?.brand_forecasts?.forEach(b => allBrandsSet.add(b.brand));

    const sortedBrands = Array.from(allBrandsSet).sort();

    sortedBrands.forEach(brand => {
      const direction = brand_insights?.brand_direction?.find(d => d.brand === brand)?.direction || 'N/A';
      const sentimentData = consumer_insights?.brand_sentiment?.find(s => s.brand === brand);
      const sentiment = sentimentData ? sentimentData.final_consumer_sentiment.toFixed(3) : 'N/A';
      const forecastInfo = forecast?.brand_forecast?.brand_forecasts?.find(f => f.brand === brand);
      const slope = forecastInfo?.trend_slope !== undefined ? forecastInfo.trend_slope.toFixed(3) : 'N/A';
      rows.push([brand.toUpperCase(), sentiment, direction, slope]);
    });

    const csvContent = rows.map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `market_report_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadPDF = () => {
    // Download the pre-generated PDF from the public folder
    const a = document.createElement('a');
    a.href = '/market_report.pdf';
    a.download = `AI_Market_Report_${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };


  const { market_overview, brand_insights } = data;
  const isPos = market_overview?.trend_slope > 0;

  return (
    <div className="animated-enter">
      <div className="page-header">
        <div>
          <h1 className="page-title">Market Overview</h1>
          <p className="text-secondary mt-1">Real-time analysis of ecommerce trends and consumer sentiment.</p>
        </div>
        <div className="page-actions">
          <select 
            className="glass-select"
            value={selectedBrand}
            onChange={e => setSelectedBrand(e.target.value)}
          >
            <option value="overall">All Brands (Market)</option>
            {brand_insights?.top_brands?.map(b => (
              <option key={b.brand} value={b.brand}>{b.brand.toUpperCase()}</option>
            ))}
          </select>
          <button
            className="glass-btn"
            onClick={handleExport}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            title="Download data as CSV spreadsheet"
          >
            <Download size={15} /> Export CSV
          </button>
          <button
            className="glass-btn primary"
            onClick={handleDownloadPDF}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            title="Download the AI-generated PDF report"
          >
            <FileText size={15} /> Download Report
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="glass-panel stat-card">
          <div className="stat-header">
            <span>Market Trend</span>
            <div className="stat-icon icon-blue">
              <Activity size={20} />
            </div>
          </div>
          <div className="stat-value">{market_overview?.trend_direction || 'Stable'}</div>
          <div className="stat-footer">
            <span className={isPos ? 'trend-up' : 'trend-down'}>
              <AnimatedNumber
                value={market_overview?.trend_slope * 100 || 0}
                formatted={`${market_overview?.trend_slope > 0 ? '+' : ''}${(market_overview?.trend_slope * 100 || 0).toFixed(1)}%`}
              />
            </span>
            <span className="text-secondary ml-1">slope intensity</span>
          </div>
        </div>

        <div className="glass-panel stat-card">
          <div className="stat-header">
            <span>Avg Consumer Sentiment</span>
            <div className="stat-icon icon-purple">
              <MessageSquare size={20} />
            </div>
          </div>
          <div className="stat-value">
            <AnimatedNumber
              value={market_overview?.current_sentiment || 0}
              formatted={(market_overview?.current_sentiment || 0).toFixed(3)}
            />
          </div>
          <div className="stat-footer">
            <span className="text-secondary">AI-calculated score from FinBERT</span>
          </div>
        </div>

        <div className="glass-panel stat-card">
          <div className="stat-header">
            <span>Market Volatility</span>
            <div className="stat-icon icon-red">
              <AlertCircle size={20} />
            </div>
          </div>
          <div className="stat-value">
            <AnimatedNumber
              value={market_overview?.volatility || 0}
              formatted={(market_overview?.volatility || 0).toFixed(2)}
            />
          </div>
          <div className="stat-footer text-secondary">
            Stability rating
          </div>
        </div>
        
        <div className="glass-panel stat-card">
          <div className="stat-header">
            <span>Top Performing Brand</span>
            <div className="stat-icon icon-green">
              <TrendingUp size={20} />
            </div>
          </div>
          <div className="stat-value capitalize">{brand_insights?.top_positive_brand || 'None'}</div>
          <div className="stat-footer text-secondary">
            Highest overall final consumer sentiment
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="charts-grid" style={{ gridTemplateColumns: '1fr' }}>
        <div className="glass-panel chart-card">
          <h3 className="chart-card-title">30-Day Forward Sentiment Forecast</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="day" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(20, 20, 30, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="sentiment" 
                  name="Sentiment Score"
                  stroke="#8b5cf6" 
                  strokeWidth={3} 
                  dot={false}
                  activeDot={{ r: 6, fill: '#8b5cf6', stroke: '#14141e', strokeWidth: 2 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
