import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { getFeatureImportance } from '../hooks/useApi';

const COLORS = [
  '#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b',
  '#10b981', '#6366f1', '#14b8a6', '#f97316', '#a855f7',
  '#0ea5e9', '#84cc16', '#e11d48', '#22d3ee', '#facc15',
];

export default function FeatureImportance() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const importance = await getFeatureImportance();
      // Take top 12 features
      setData(importance.slice(0, 12).reverse());
    } catch (err) {
      console.error('Failed to load feature importance:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="loading"><span className="spinner" /> Loading feature importance...</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__title">
        <span className="card__title-icon">📊</span>
        Global Feature Importance (SHAP)
      </div>

      {data.length > 0 ? (
        <div className="chart-container--tall">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,160,255,0.08)" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fontSize: 10 }}
                label={{ value: 'Mean |SHAP|', position: 'bottom', fill: '#64748b', fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                width={160}
              />
              <Tooltip
                contentStyle={{
                  background: '#0c1322',
                  border: '1px solid rgba(100,160,255,0.15)',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                }}
                formatter={(value) => [value.toFixed(4), 'Importance']}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
                {data.map((entry, index) => (
                  <Cell key={entry.feature} fill={COLORS[index % COLORS.length]} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state__icon">📊</div>
          <p>No feature importance data available</p>
        </div>
      )}
    </div>
  );
}
