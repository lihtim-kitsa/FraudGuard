import { useState } from 'react';

export default function MetricsOverview({ metrics, health }) {
  const items = [
    {
      label: 'PR-AUC',
      value: metrics?.pr_auc?.toFixed(4) || '—',
      style: 'accent',
    },
    {
      label: 'Precision',
      value: metrics?.precision?.toFixed(4) || '—',
      style: '',
    },
    {
      label: 'Recall',
      value: metrics?.recall?.toFixed(4) || '—',
      style: '',
    },
    {
      label: 'F1 Score',
      value: metrics?.f1?.toFixed(4) || '—',
      style: '',
    },
    {
      label: 'ROC-AUC',
      value: metrics?.roc_auc?.toFixed(4) || '—',
      style: '',
    },
    {
      label: 'Threshold',
      value: metrics?.threshold?.toFixed(3) || '—',
      style: 'review',
    },
    {
      label: 'Total Scored',
      value: metrics?.total_predictions?.toLocaleString() || '0',
      style: '',
    },
    {
      label: 'Fraud Rate',
      value: metrics?.fraud_rate ? (metrics.fraud_rate * 100).toFixed(2) + '%' : '—',
      style: 'decline',
    },
  ];

  return (
    <div className="grid-metrics">
      {items.map((item) => (
        <div className="metric-card" key={item.label}>
          <div className="metric-card__label">{item.label}</div>
          <div className={`metric-card__value ${item.style ? `metric-card__value--${item.style}` : ''}`}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}
