/**
 * API integration hooks for FraudGuard dashboard.
 */

const API_BASE = '';  // Proxied via Vite

export async function scoreTransaction(transaction) {
  const res = await fetch(`${API_BASE}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });
  if (!res.ok) throw new Error(`Score failed: ${res.status}`);
  return res.json();
}

export async function getThreshold() {
  const res = await fetch(`${API_BASE}/threshold`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function updateThreshold(declineThreshold, reviewThreshold) {
  const res = await fetch(`${API_BASE}/threshold`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      decline_threshold: declineThreshold,
      review_threshold: reviewThreshold,
    }),
  });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getThresholdSweep() {
  const res = await fetch(`${API_BASE}/threshold/sweep`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/monitoring/metrics`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getFeatureImportance() {
  const res = await fetch(`${API_BASE}/monitoring/feature-importance`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getDrift() {
  const res = await fetch(`${API_BASE}/monitoring/drift`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getRecentPredictions() {
  const res = await fetch(`${API_BASE}/monitoring/predictions/recent`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}
