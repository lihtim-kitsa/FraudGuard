import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { getThresholdSweep, updateThreshold } from '../hooks/useApi';

export default function ThresholdSlider() {
  const [threshold, setThreshold] = useState(0.5);
  const [reviewThreshold, setReviewThreshold] = useState(0.3);
  const [sweepData, setSweepData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSweep();
  }, []);

  const loadSweep = async () => {
    try {
      const data = await getThresholdSweep();
      setSweepData(data);
      // Find initial metrics at default threshold
      const closest = data.reduce((prev, curr) =>
        Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
      );
      setMetrics(closest);
    } catch (err) {
      console.error('Failed to load sweep data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleThresholdChange = (e) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);

    // Find closest point in sweep data
    if (sweepData.length > 0) {
      const closest = sweepData.reduce((prev, curr) =>
        Math.abs(curr.threshold - val) < Math.abs(prev.threshold - val) ? curr : prev
      );
      setMetrics(closest);
    }
  };

  const handleApply = async () => {
    try {
      const res = await updateThreshold(threshold, Math.min(reviewThreshold, threshold - 0.01));
      setMetrics(res);
    } catch (err) {
      console.error('Failed to apply threshold:', err);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="loading"><span className="spinner" /> Loading threshold data...</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__title">
        <span className="card__title-icon">🎚️</span>
        Decision Threshold Tuning
      </div>

      <div className="slider-container">
        <div className="slider-label">
          <span className="slider-label__text">Decline Threshold</span>
          <span className="slider-label__value">{threshold.toFixed(2)}</span>
        </div>
        <input
          id="threshold-slider"
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          value={threshold}
          onChange={handleThresholdChange}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          <span>← More approvals</span>
          <span>More declines →</span>
        </div>
      </div>

      {metrics && (
        <div className="grid-4" style={{ marginTop: '16px', marginBottom: '16px' }}>
          <div className="metric-card">
            <div className="metric-card__label">Precision</div>
            <div className="metric-card__value">{(metrics.precision || 0).toFixed(3)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-card__label">Recall</div>
            <div className="metric-card__value">{(metrics.recall || 0).toFixed(3)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-card__label">F1</div>
            <div className="metric-card__value">{(metrics.f1 || 0).toFixed(3)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-card__label">Business Cost</div>
            <div className="metric-card__value metric-card__value--decline">
              ${(metrics.total_cost || 0).toLocaleString()}
            </div>
          </div>
        </div>
      )}

      <button className="btn btn--primary btn--full" onClick={handleApply} id="btn-apply-threshold">
        Apply Threshold
      </button>

      {sweepData.length > 0 && (
        <div className="chart-container" style={{ marginTop: '24px' }}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={sweepData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,160,255,0.08)" />
              <XAxis
                dataKey="threshold"
                label={{ value: 'Threshold', position: 'bottom', fill: '#64748b', fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  background: '#0c1322',
                  border: '1px solid rgba(100,160,255,0.15)',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
              <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={2} dot={false} name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#06b6d4" strokeWidth={2} dot={false} name="Recall" />
              <Line type="monotone" dataKey="f1" stroke="#8b5cf6" strokeWidth={2} dot={false} name="F1" />
              <ReferenceLine x={threshold} stroke="#f59e0b" strokeDasharray="5 5" strokeWidth={2} label="" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
