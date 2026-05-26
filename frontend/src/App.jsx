import { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/DashboardLayout';
import Overview from './pages/Overview';
import Topics from './pages/Topics';
import Comparison from './pages/Comparison';
import Alerts from './pages/Alerts';
import Sources from './pages/Sources';
import News from './pages/News';
import Chatbot from './components/Chatbot';

function App() {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isError, setIsError] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);

  const fetchRealTimeData = useCallback(async () => {
    // 1. Pause fetching when the tab is hidden to save resources
    if (document.hidden) return;

    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ""; // Defaults to empty for local/Vercel static serving
    try {
      const endpoint = API_BASE_URL ? `${API_BASE_URL}/dashboard-data` : '/market_dashboard_data.json';
      const res = await fetch(endpoint + '?t=' + new Date().getTime());
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      setMarketData(data);
      setLastUpdated(new Date());
      setIsError(false);
    } catch (err) {
      console.error("Failed to load dashboard data", err);
      setIsError(true);
    } finally {
      setIsSyncing(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchRealTimeData();

    // Set up Polling: Fetch new data every 10 seconds (10000 milliseconds)
    const intervalId = setInterval(fetchRealTimeData, 10000);

    // 2. Add event listener to fetch immediately when user returns to tab
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        fetchRealTimeData();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Cleanup the interval & event listener if the component unmounts
    return () => {
      clearInterval(intervalId);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [fetchRealTimeData]);

  if (loading && !marketData) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#e2e8f0', background: '#0a0a0f' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '40px', height: '40px', border: '3px solid rgba(255,255,255,0.1)', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          <div className="text-xl font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Initializing AI Pipeline...
          </div>
        </div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout data={marketData} lastUpdated={lastUpdated} isError={isError} isSyncing={isSyncing} onRefresh={fetchRealTimeData} />}>
          <Route index element={<Overview data={marketData} />} />
          <Route path="topics" element={<Topics data={marketData} />} />
          <Route path="comparison" element={<Comparison data={marketData} />} />
          <Route path="alerts" element={<Alerts data={marketData} />} />
          <Route path="news" element={<News data={marketData} />} />
          <Route path="sources" element={<Sources data={marketData} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <Chatbot data={marketData} />
    </BrowserRouter>
  );
}

export default App;
