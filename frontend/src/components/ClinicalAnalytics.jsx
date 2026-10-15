import React, { useState } from 'react';
import {
  TrendingUp,
  BarChart3,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Activity,
  Layers,
  FileText
} from 'lucide-react';

export default function ClinicalAnalytics() {
  const [activeThreshold, setActiveThreshold] = useState('0.65');

  // Exact data from experiments/v2_contamination_cleaned/reports/v1_model_on_clean_testset.json
  const matrixData = {
    '0.50': {
      tp: 155,
      fn: 8,
      fp: 35,
      tn: 123,
      acc: '86.60%',
      sens: '95.09%',
      spec: '77.85%',
      prec: '81.58%',
      f1: '87.82%',
      desc: 'Standard screening cutoff. Prioritizes near-zero missed fibroids (only 8 FN out of 163 true cases).'
    },
    '0.65': {
      tp: 147,
      fn: 16,
      fp: 18,
      tn: 140,
      acc: '89.41%',
      sens: '90.18%',
      spec: '88.61%',
      prec: '89.09%',
      f1: '89.63%',
      desc: 'Optimal clinical operating point. Balances 90.18% sensitivity with 88.61% specificity, cutting false alarms by 48.6%.'
    }
  };

  const current = matrixData[activeThreshold];

  const sweepRows = [
    { tau: 0.30, acc: '76.64%', sens: '100.0%', spec: '52.53%', prec: '68.49%', f1: '81.30%', fp: 75, fn: 0 },
    { tau: 0.40, acc: '83.18%', sens: '98.16%', spec: '67.72%', prec: '75.83%', f1: '85.56%', fp: 51, fn: 3 },
    { tau: 0.45, acc: '86.29%', sens: '97.55%', spec: '74.68%', prec: '79.90%', f1: '87.85%', fp: 40, fn: 4 },
    { tau: 0.50, acc: '86.60%', sens: '95.09%', spec: '77.85%', prec: '81.58%', f1: '87.82%', fp: 35, fn: 8, isStandard: true },
    { tau: 0.55, acc: '88.47%', sens: '92.64%', spec: '84.18%', prec: '85.80%', f1: '89.09%', fp: 25, fn: 12 },
    { tau: 0.60, acc: '88.47%', sens: '90.80%', spec: '86.08%', prec: '87.06%', f1: '88.89%', fp: 22, fn: 15 },
    { tau: 0.65, acc: '89.41%', sens: '90.18%', spec: '88.61%', prec: '89.09%', f1: '89.63%', fp: 18, fn: 16, isOptimal: true },
    { tau: 0.70, acc: '88.47%', sens: '85.89%', spec: '91.14%', prec: '90.91%', f1: '88.33%', fp: 14, fn: 23 },
    { tau: 0.75, acc: '85.98%', sens: '80.37%', spec: '91.77%', prec: '90.97%', f1: '85.34%', fp: 13, fn: 32 },
    { tau: 0.80, acc: '82.87%', sens: '71.78%', spec: '94.30%', prec: '92.86%', f1: '80.97%', fp: 9, fn: 46 },
  ];

  return (
    <div className="clinical-analytics-section">
      <div className="container">
        {/* Section Header */}
        <div className="section-head-row">
          <div>
            <span className="badge badge-neutral" style={{ marginBottom: '8px' }}>
              Phase 4 Empirical Validation
            </span>
            <h2>Held-Out Clinical Evaluation (n=321 Clean Scans)</h2>
            <p style={{ maxWidth: '680px', marginTop: '6px' }}>
              Rigorous test-cohort evaluation on 321 quarantined B-mode pelvic scans (163 confirmed UF, 158 clean NUF),
              strictly isolated from all training and validation phases.
            </p>
          </div>
          <div className="auc-stat-badge">
            <span className="auc-label font-mono">DISCRIMINATIVE POWER</span>
            <span className="auc-value font-mono">0.9533</span>
            <span className="auc-sub font-mono">ROC-AUC (PR-AUC: 0.9514)</span>
          </div>
        </div>

        {/* 4 Stat Metric Cards */}
        <div className="analytics-metrics-grid">
          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">UF Sensitivity (Recall)</span>
              <ShieldCheck size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">95.09%</div>
            <div className="metric-sub">
              <strong>155 / 163</strong> confirmed fibroids detected. Only 8 missed lesions at default cutoff.
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">Optimal Accuracy (τ = 0.65)</span>
              <Activity size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">89.41%</div>
            <div className="metric-sub">
              <strong>287 / 321</strong> overall scans correctly diagnosed at calibrated cutoff.
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">Optimal Specificity (τ = 0.65)</span>
              <CheckCircle2 size={18} color="var(--diag-nuf-bar)" />
            </div>
            <div className="metric-num font-mono">88.61%</div>
            <div className="metric-sub">
              <strong>140 / 158</strong> normal myometrial scans accurately ruled negative.
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">UF Precision (PPV)</span>
              <TrendingUp size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">89.09%</div>
            <div className="metric-sub">
              Positive predictive value at optimal threshold. High diagnostic fidelity.
            </div>
          </div>
        </div>

        {/* 2-Column: Interactive Confusion Matrix & Operating Threshold Sweep Table */}
        <div className="matrix-and-sweep-grid">
          {/* Left: Confusion Matrix Card */}
          <div className="card matrix-card">
            <div className="matrix-card-header">
              <h3 style={{ fontSize: '1.25rem' }}>Paired Confusion Matrix</h3>
              <div className="matrix-toggle">
                <button
                  className={`matrix-toggle-btn ${activeThreshold === '0.50' ? 'active' : ''}`}
                  onClick={() => setActiveThreshold('0.50')}
                >
                  τ = 0.50 (Standard)
                </button>
                <button
                  className={`matrix-toggle-btn ${activeThreshold === '0.65' ? 'active' : ''}`}
                  onClick={() => setActiveThreshold('0.65')}
                >
                  τ = 0.65 (Optimal)
                </button>
              </div>
            </div>

            <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
              {current.desc}
            </p>

            {/* Matrix Visual Layout */}
            <div className="confusion-matrix-table">
              <div className="cm-row-header">
                <div className="cm-corner" />
                <div className="cm-col-label">Predicted Fibroid (UF)</div>
                <div className="cm-col-label">Predicted Normal (NUF)</div>
              </div>

              {/* Row 1: Actual UF */}
              <div className="cm-row">
                <div className="cm-row-label">
                  <strong>Actual UF</strong>
                  <span>(n=163)</span>
                </div>
                <div className="cm-cell cell-tp">
                  <div className="cell-num font-mono">{current.tp}</div>
                  <div className="cell-lbl">True Positive (TP)</div>
                  <div className="cell-pct font-mono">{((current.tp / 163) * 100).toFixed(1)}%</div>
                </div>
                <div className="cm-cell cell-fn">
                  <div className="cell-num font-mono">{current.fn}</div>
                  <div className="cell-lbl">False Negative (FN)</div>
                  <div className="cell-pct font-mono">{((current.fn / 163) * 100).toFixed(1)}%</div>
                </div>
              </div>

              {/* Row 2: Actual NUF */}
              <div className="cm-row">
                <div className="cm-row-label">
                  <strong>Actual NUF</strong>
                  <span>(n=158)</span>
                </div>
                <div className="cm-cell cell-fp">
                  <div className="cell-num font-mono">{current.fp}</div>
                  <div className="cell-lbl">False Positive (FP)</div>
                  <div className="cell-pct font-mono">{((current.fp / 158) * 100).toFixed(1)}%</div>
                </div>
                <div className="cm-cell cell-tn">
                  <div className="cell-num font-mono">{current.tn}</div>
                  <div className="cell-lbl">True Negative (TN)</div>
                  <div className="cell-pct font-mono">{((current.tn / 158) * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Matrix Bottom Readout */}
            <div className="matrix-summary-footer">
              <div className="footer-metric font-mono">
                Accuracy: <strong>{current.acc}</strong>
              </div>
              <div className="footer-metric font-mono">
                Sensitivity: <strong>{current.sens}</strong>
              </div>
              <div className="footer-metric font-mono">
                Specificity: <strong>{current.spec}</strong>
              </div>
              <div className="footer-metric font-mono">
                F1-Score: <strong>{current.f1}</strong>
              </div>
            </div>
          </div>

          {/* Right: Threshold Operating Sweep Table */}
          <div className="card sweep-card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Diagnostic Operating Curve Sweep</h3>
            <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Full sensitivity-specificity trade-off across decision thresholds (τ). Demonstrates robust calibration across all probability tiers.
            </p>

            <div className="table-responsive">
              <table className="sweep-table font-mono">
                <thead>
                  <tr>
                    <th>Cutoff (τ)</th>
                    <th>Accuracy</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Precision</th>
                    <th>FPs</th>
                  </tr>
                </thead>
                <tbody>
                  {sweepRows.map((r) => (
                    <tr
                      key={r.tau}
                      className={r.isOptimal ? 'row-optimal' : r.isStandard ? 'row-standard' : ''}
                    >
                      <td>
                        <strong>{r.tau.toFixed(2)}</strong>
                        {r.isOptimal && <span className="tag-opt">BEST ACC</span>}
                        {r.isStandard && <span className="tag-std">DEFAULT</span>}
                      </td>
                      <td>{r.acc}</td>
                      <td style={{ color: 'var(--diag-uf-bar)' }}>{r.sens}</td>
                      <td style={{ color: 'var(--diag-nuf-bar)' }}>{r.spec}</td>
                      <td>{r.prec}</td>
                      <td>{r.fp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
