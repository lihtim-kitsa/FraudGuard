import { useState, useEffect } from 'react';
import { getThresholdSweep } from '../hooks/useApi';

export default function ConfusionMatrix({ initialThreshold = 0.5 }) {
  const [threshold, setThreshold] = useState(initialThreshold);
  const [sweepData, setSweepData] = useState([]);
  const [matrix, setMatrix] = useState({ tp: 0, fp: 0, tn: 0, fn: 0 });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const data = await getThresholdSweep();
      setSweepData(data);
      updateMatrix(data, threshold);
    } catch (err) {
      console.error('Failed to load data:', err);
    }
  };

  const updateMatrix = (data, t) => {
    if (data.length === 0) return;
    const closest = data.reduce((prev, curr) =>
      Math.abs(curr.threshold - t) < Math.abs(prev.threshold - t) ? curr : prev
    );
    setMatrix({
      tp: closest.tp || 0,
      fp: closest.fp || 0,
      tn: closest.tn || 0,
      fn: closest.fn || 0,
    });
  };

  const handleSlider = (e) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);
    updateMatrix(sweepData, val);
  };

  const total = matrix.tp + matrix.fp + matrix.tn + matrix.fn || 1;

  return (
    <div className="card">
      <div className="card__title">
        <span className="card__title-icon">📐</span>
        Confusion Matrix
      </div>

      <div className="slider-container">
        <div className="slider-label">
          <span className="slider-label__text">Threshold</span>
          <span className="slider-label__value">{threshold.toFixed(2)}</span>
        </div>
        <input
          id="cm-threshold-slider"
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          value={threshold}
          onChange={handleSlider}
        />
      </div>

      <div className="confusion-matrix">
        {/* Header row */}
        <div className="cm-header"></div>
        <div className="cm-header">Predicted Legit</div>
        <div className="cm-header">Predicted Fraud</div>

        {/* Row 1: Actual Legit */}
        <div className="cm-header" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
          Actual Legit
        </div>
        <div className="cm-cell cm-cell--tn">
          <div className="cm-cell__value">{matrix.tn.toLocaleString()}</div>
          <div className="cm-cell__label">True Neg</div>
        </div>
        <div className="cm-cell cm-cell--fp">
          <div className="cm-cell__value">{matrix.fp.toLocaleString()}</div>
          <div className="cm-cell__label">False Pos</div>
        </div>

        {/* Row 2: Actual Fraud */}
        <div className="cm-header" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
          Actual Fraud
        </div>
        <div className="cm-cell cm-cell--fn">
          <div className="cm-cell__value">{matrix.fn.toLocaleString()}</div>
          <div className="cm-cell__label">False Neg</div>
        </div>
        <div className="cm-cell cm-cell--tp">
          <div className="cm-cell__value">{matrix.tp.toLocaleString()}</div>
          <div className="cm-cell__label">True Pos</div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginTop: '16px' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Accuracy: {((matrix.tp + matrix.tn) / total * 100).toFixed(1)}% •
          Error rate: {((matrix.fp + matrix.fn) / total * 100).toFixed(2)}%
        </div>
      </div>
    </div>
  );
}
