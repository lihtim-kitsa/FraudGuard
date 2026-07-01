import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import { getDrift } from '../hooks/useApi';

const FEATURE_COLORS = {
  amount: '#3b82f6',
  distance_from_home: '#06b6d4',
  velocity_24h: '#8b5cf6',
  device_trust_score: '#f59e0b',
  hour_of_day: '#ec4899',
};

export default function DriftMonitor() {
  const [driftData, setDriftData] = useState([]);
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const raw = await getDrift();

      // Pivot data: group by window, features become columns
      const windowMap = {};
      const featureSet = new Set();

      raw.forEach((point) => {
        featureSet.add(point.feature);
        if (!windowMap[point.window]) {
          windowMap[point.window] = { window: point.window };
        }
        windowMap[point.window][point.feature] = point.drift_score;
      });

      setFeatures([...featureSet]);
      setDriftData(Object.values(windowMap));
    } catch (err) {
      console.error('Failed to load drift data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="loading"><span className="spinner" /> Loading drift data...</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__title">
        <span className="card__title-icon">📈</span>
        Feature Drift Monitor
      </div>

      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
        Tracks feature distribution shift over time windows.
        High drift scores indicate the model may need retraining.
      </div>

      {driftData.length > 0 ? (
        <div className="chart-container--tall">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={driftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,160,255,0.08)" />
              <XAxis dataKey="window" tick={{ fontSize: 10 }} />
              <YAxis
                tick={{ fontSize: 10 }}
                label={{ value: 'Drift Score', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  background: '#0c1322',
                  border: '1px solid rgba(100,160,255,0.15)',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                }}
                formatter={(value) => [value?.toFixed(4) ?? '—', '']}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
              {features.map((feature) => (
                <Line
                  key={feature}
                  type="monotone"
                  dataKey={feature}
                  stroke={FEATURE_COLORS[feature] || '#94a3b8'}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  name={feature.replace(/_/g, ' ')}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state__icon">📈</div>
          <p>No drift data available</p>
        </div>
      )}

      <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(245,158,11,0.08)', borderRadius: '8px', border: '1px solid rgba(245,158,11,0.15)' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--color-review)', fontWeight: 600 }}>
          ⚠ Drift Detection Note
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          This is simulated drift data for demonstration.
          In production, compare live feature distributions against training data baselines using
          PSI (Population Stability Index) or KS tests.
        </div>
      </div>
    </div>
  );
}
