import { useState, useEffect } from 'react';
import MetricsOverview from './components/MetricsOverview';
import ScoreTransaction from './components/ScoreTransaction';
import ThresholdSlider from './components/ThresholdSlider';
import ConfusionMatrix from './components/ConfusionMatrix';
import FeatureImportance from './components/FeatureImportance';
import DriftMonitor from './components/DriftMonitor';
import { getMetrics, getHealth } from './hooks/useApi';

const TABS = [
  { id: 'scoring', label: '⚡ Live Scoring', icon: '⚡' },
  { id: 'threshold', label: '🎚️ Threshold Tuning', icon: '🎚️' },
  { id: 'monitoring', label: '📊 Model Monitoring', icon: '📊' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('scoring');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
    // Poll health every 30s
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    await loadHealth();
    try {
      const m = await getMetrics();
      setMetrics(m);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  };

  const loadHealth = async () => {
    try {
      const h = await getHealth();
      setHealth(h);
      setError(null);
    } catch (err) {
      setError('API unavailable');
      setHealth(null);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__logo">
          <div className="header__icon">🛡️</div>
          <div>
            <div className="header__title">FraudGuard</div>
            <div className="header__subtitle">Real-Time Fraud Detection System</div>
          </div>
        </div>
        <div className="header__status">
          <span className={`status-dot ${error ? 'status-dot--error' : ''}`} />
          {error ? error : health?.model_loaded ? `${health.model_type} loaded` : 'Connecting...'}
        </div>
      </header>

      {/* KPI Metrics */}
      <MetricsOverview metrics={metrics} health={health} />

      {/* Navigation */}
      <nav className="nav-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            className={`nav-tab ${activeTab === tab.id ? 'nav-tab--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Tab Content */}
      {activeTab === 'scoring' && (
        <ScoreTransaction />
      )}

      {activeTab === 'threshold' && (
        <div>
          <div className="grid-2">
            <ThresholdSlider />
            <ConfusionMatrix />
          </div>
        </div>
      )}

      {activeTab === 'monitoring' && (
        <div>
          <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
            <FeatureImportance />
            <DriftMonitor />
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '32px 0 16px',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        borderTop: '1px solid var(--glass-border)',
        marginTop: '48px',
      }}>
        FraudGuard v1.0.0 • Built with FastAPI + XGBoost + SHAP + React
      </footer>
    </div>
  );
}
