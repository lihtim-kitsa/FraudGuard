import { useState } from 'react';
import { scoreTransaction } from '../hooks/useApi';

const MERCHANT_CATEGORIES = [
  'grocery', 'gas_station', 'restaurant', 'online_retail',
  'travel', 'entertainment', 'healthcare', 'electronics',
  'jewelry', 'cash_advance',
];

const DEFAULT_VALUES = {
  amount: 245.50,
  merchant_category: 'electronics',
  hour_of_day: 2.5,
  day_of_week: 5,
  is_foreign: 1,
  distance_from_home: 120,
  device_trust_score: 0.3,
};

export default function ScoreTransaction() {
  const [form, setForm] = useState(DEFAULT_VALUES);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...form,
        amount: parseFloat(form.amount),
        hour_of_day: parseFloat(form.hour_of_day),
        day_of_week: parseInt(form.day_of_week),
        is_foreign: parseInt(form.is_foreign),
        distance_from_home: parseFloat(form.distance_from_home),
        device_trust_score: parseFloat(form.device_trust_score),
      };
      const res = await scoreTransaction(payload);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRandomFraud = () => {
    setForm({
      amount: (Math.random() * 2000 + 500).toFixed(2),
      merchant_category: ['cash_advance', 'jewelry', 'electronics'][Math.floor(Math.random() * 3)],
      hour_of_day: (Math.random() * 5 + 1).toFixed(1),
      day_of_week: Math.floor(Math.random() * 7),
      is_foreign: 1,
      distance_from_home: (Math.random() * 200 + 50).toFixed(0),
      device_trust_score: (Math.random() * 0.3).toFixed(2),
    });
  };

  const handleRandomLegit = () => {
    setForm({
      amount: (Math.random() * 80 + 10).toFixed(2),
      merchant_category: ['grocery', 'gas_station', 'restaurant'][Math.floor(Math.random() * 3)],
      hour_of_day: (Math.random() * 8 + 9).toFixed(1),
      day_of_week: Math.floor(Math.random() * 5),
      is_foreign: 0,
      distance_from_home: (Math.random() * 15 + 1).toFixed(0),
      device_trust_score: (Math.random() * 0.15 + 0.85).toFixed(2),
    });
  };

  const probabilityColor = (prob) => {
    if (prob >= 0.7) return 'var(--color-decline)';
    if (prob >= 0.3) return 'var(--color-review)';
    return 'var(--color-approve)';
  };

  return (
    <div className="grid-2">
      <div className="card">
        <div className="card__title">
          <span className="card__title-icon">🔍</span>
          Score Transaction
        </div>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <button className="btn btn--secondary" onClick={handleRandomFraud} type="button" id="btn-random-fraud">
            🎲 Random Fraud
          </button>
          <button className="btn btn--secondary" onClick={handleRandomLegit} type="button" id="btn-random-legit">
            🎲 Random Legit
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Amount ($)</label>
              <input
                id="input-amount"
                className="form-input"
                type="number"
                step="0.01"
                value={form.amount}
                onChange={(e) => handleChange('amount', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Merchant Category</label>
              <select
                id="input-merchant"
                className="form-select"
                value={form.merchant_category}
                onChange={(e) => handleChange('merchant_category', e.target.value)}
              >
                {MERCHANT_CATEGORIES.map((cat) => (
                  <option key={cat} value={cat}>{cat.replace('_', ' ')}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Hour of Day (0-24)</label>
              <input
                id="input-hour"
                className="form-input"
                type="number"
                step="0.1"
                min="0"
                max="23.9"
                value={form.hour_of_day}
                onChange={(e) => handleChange('hour_of_day', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Day of Week (0-6)</label>
              <input
                id="input-day"
                className="form-input"
                type="number"
                min="0"
                max="6"
                value={form.day_of_week}
                onChange={(e) => handleChange('day_of_week', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Foreign Transaction</label>
              <select
                id="input-foreign"
                className="form-select"
                value={form.is_foreign}
                onChange={(e) => handleChange('is_foreign', e.target.value)}
              >
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Distance from Home (km)</label>
              <input
                id="input-distance"
                className="form-input"
                type="number"
                min="0"
                value={form.distance_from_home}
                onChange={(e) => handleChange('distance_from_home', e.target.value)}
              />
            </div>
            <div className="form-group" style={{ gridColumn: 'span 2' }}>
              <label className="form-label">Device Trust Score (0-1)</label>
              <input
                id="input-trust"
                className="form-input"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={form.device_trust_score}
                onChange={(e) => handleChange('device_trust_score', e.target.value)}
              />
            </div>
          </div>

          <button
            className="btn btn--primary btn--full"
            type="submit"
            disabled={loading}
            id="btn-score"
          >
            {loading ? (
              <><span className="spinner" /> Scoring...</>
            ) : (
              '⚡ Score Transaction'
            )}
          </button>
        </form>

        {error && (
          <div style={{ marginTop: '16px', padding: '12px', background: 'var(--color-decline-bg)', borderRadius: '8px', color: 'var(--color-decline)', fontSize: '0.85rem' }}>
            ⚠ {error}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card__title">
          <span className="card__title-icon">📊</span>
          Scoring Result
        </div>

        {result ? (
          <div className="result-panel">
            <div className="result-header">
              <div>
                <div className="result-probability" style={{ color: probabilityColor(result.fraud_probability) }}>
                  {(result.fraud_probability * 100).toFixed(1)}%
                </div>
                <div className="result-latency">
                  {result.latency_ms.toFixed(1)}ms • ID: {result.transaction_id}
                </div>
              </div>
              <span className={`decision-badge decision-badge--${result.decision}`}>
                {result.decision === 'approve' && '✓'}
                {result.decision === 'review' && '⚠'}
                {result.decision === 'decline' && '✕'}
                {' '}{result.decision}
              </span>
            </div>

            <div className="explanation-title">Why this decision</div>
            <div className="explanation-list">
              {result.explanation?.map((feat, i) => (
                <div
                  key={i}
                  className={`explanation-item ${feat.shap_value > 0 ? 'explanation-item--positive' : 'explanation-item--negative'}`}
                >
                  <span className="explanation-feature">{feat.feature}</span>
                  <span className={`explanation-value ${feat.shap_value > 0 ? 'explanation-value--positive' : 'explanation-value--negative'}`}>
                    {feat.shap_value > 0 ? '+' : ''}{feat.shap_value.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state__icon">🛡️</div>
            <p>Submit a transaction to see fraud scoring results with SHAP explanations</p>
          </div>
        )}
      </div>
    </div>
  );
}
