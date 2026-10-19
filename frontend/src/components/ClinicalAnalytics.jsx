import React, { useState } from 'react';
import {
  TrendingUp,
  ShieldCheck,
  CheckCircle2,
  Activity
} from 'lucide-react';

export default function ClinicalAnalytics() {
  const [activeRule, setActiveRule] = useState('rule_2');

  // Exact data from recovered_e0_report.json (1,474 development scans across 917 study groups)
  const matrixData = {
    'rule_1': {
      tp: 646,
      fn: 39,
      fp: 55,
      tn: 734,
      acc: '93.65% ± 0.74%',
      sens: '94.31%',
      spec: '93.10%',
      prec: '92.25%',
      f1: '0.9321',
      roc_auc: '0.9847',
      desc: 'Rule 1: Minimum Inner-Validation Loss Checkpoints. Prioritizes low entropy log-loss during model selection.'
    },
    'rule_2': {
      tp: 649,
      fn: 36,
      fp: 51,
      tn: 738,
      acc: '94.12% ± 0.64%',
      sens: '94.75%',
      spec: '93.57%',
      prec: '92.78%',
      f1: '0.9372',
      roc_auc: '0.9843',
      desc: 'Rule 2: Maximum Inner-Validation Accuracy Checkpoints. Authoritative production baseline benchmark.'
    }
  };

  const current = matrixData[activeRule];

  // Per-fold outer validation details from recovered_e0_report.json
  const foldRows = [
    { fold: 'Fold 0', acc: '94.04%', sens: '92.59%', spec: '95.21%', prec: '93.98%', f1: '0.9328', auc: '0.9804', ep: 3 },
    { fold: 'Fold 1', acc: '94.29%', sens: '97.04%', spec: '91.72%', prec: '91.61%', f1: '0.9424', auc: '0.9867', ep: 5 },
    { fold: 'Fold 2', acc: '92.93%', sens: '95.65%', spec: '90.75%', prec: '89.19%', f1: '0.9231', auc: '0.9802', ep: 5 },
    { fold: 'Fold 3', acc: '94.56%', sens: '94.12%', spec: '94.94%', prec: '94.12%', f1: '0.9412', auc: '0.9838', ep: 5 },
    { fold: 'Fold 4', acc: '94.77%', sens: '94.33%', spec: '95.21%', prec: '95.00%', f1: '0.9466', auc: '0.9904', ep: 6 },
  ];

  return (
    <div className="clinical-analytics-section">
      <div className="container">
        {/* Section Header */}
        <div className="section-head-row">
          <div>
            <span className="badge badge-neutral" style={{ marginBottom: '8px' }}>
              Authoritative Development Benchmark
            </span>
            <h2>5-Fold Nested Grouped Cross-Validation (n=1,474 Scans)</h2>
            <p style={{ maxWidth: '680px', marginTop: '6px' }}>
              Rigorous nested evaluation on the recovered cohort of 1,474 pelvic ultrasound scans (685 confirmed UF, 789 clean NUF)
              partitioned across 917 acquisition-study proxy groups. Checkpoints selected strictly via inner folds.
            </p>
          </div>
          <div className="auc-stat-badge">
            <span className="auc-label font-mono">DISCRIMINATIVE POWER</span>
            <span className="auc-value font-mono">0.9843</span>
            <span className="auc-sub font-mono">Mean Outer ROC-AUC (PR-AUC: 0.9822)</span>
          </div>
        </div>

        {/* 4 Stat Metric Cards */}
        <div className="analytics-metrics-grid">
          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">Mean Sensitivity (Recall)</span>
              <ShieldCheck size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">94.75%</div>
            <div className="metric-sub">
              <strong>649 / 685</strong> pooled true fibroids detected across all 5 outer folds (±1.50% std).
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">Outer Fold Accuracy</span>
              <Activity size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">94.12%</div>
            <div className="metric-sub">
              Mean outer accuracy across folds (<strong>±0.64% std</strong>; pooled accuracy 94.10%).
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">Mean Specificity</span>
              <CheckCircle2 size={18} color="var(--diag-nuf-bar)" />
            </div>
            <div className="metric-num font-mono">93.57%</div>
            <div className="metric-sub">
              <strong>738 / 789</strong> normal myometrial scans accurately ruled negative (±1.93% std).
            </div>
          </div>

          <div className="card metric-card">
            <div className="metric-header">
              <span className="metric-title">F1-Score / Precision</span>
              <TrendingUp size={18} color="var(--accent-primary)" />
            </div>
            <div className="metric-num font-mono">0.9372</div>
            <div className="metric-sub">
              Mean F1 of <strong>0.9372</strong> with high precision of <strong>92.78%</strong> on positive calls.
            </div>
          </div>
        </div>

        {/* 2-Column: Interactive Confusion Matrix & Outer Fold Table */}
        <div className="matrix-and-sweep-grid">
          {/* Left: Confusion Matrix Card */}
          <div className="card matrix-card">
            <div className="matrix-card-header">
              <h3 style={{ fontSize: '1.25rem' }}>Pooled Confusion Matrix (n=1,474)</h3>
              <div className="matrix-toggle">
                <button
                  className={`matrix-toggle-btn ${activeRule === 'rule_2' ? 'active' : ''}`}
                  onClick={() => setActiveRule('rule_2')}
                >
                  Rule 2 (Max Acc)
                </button>
                <button
                  className={`matrix-toggle-btn ${activeRule === 'rule_1' ? 'active' : ''}`}
                  onClick={() => setActiveRule('rule_1')}
                >
                  Rule 1 (Min Loss)
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
                  <span>(n=685)</span>
                </div>
                <div className="cm-cell cell-tp">
                  <div className="cell-num font-mono">{current.tp}</div>
                  <div className="cell-lbl">True Positive (TP)</div>
                  <div className="cell-pct font-mono">{((current.tp / 685) * 100).toFixed(1)}%</div>
                </div>
                <div className="cm-cell cell-fn">
                  <div className="cell-num font-mono">{current.fn}</div>
                  <div className="cell-lbl">False Negative (FN)</div>
                  <div className="cell-pct font-mono">{((current.fn / 685) * 100).toFixed(1)}%</div>
                </div>
              </div>

              {/* Row 2: Actual NUF */}
              <div className="cm-row">
                <div className="cm-row-label">
                  <strong>Actual NUF</strong>
                  <span>(n=789)</span>
                </div>
                <div className="cm-cell cell-fp">
                  <div className="cell-num font-mono">{current.fp}</div>
                  <div className="cell-lbl">False Positive (FP)</div>
                  <div className="cell-pct font-mono">{((current.fp / 789) * 100).toFixed(1)}%</div>
                </div>
                <div className="cm-cell cell-tn">
                  <div className="cell-num font-mono">{current.tn}</div>
                  <div className="cell-lbl">True Negative (TN)</div>
                  <div className="cell-pct font-mono">{((current.tn / 789) * 100).toFixed(1)}%</div>
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

          {/* Right: Per-Fold Outer Validation Table */}
          <div className="card sweep-card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Per-Fold Outer Validation Performance</h3>
            <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Each outer fold was evaluated strictly once on unseen study groups. Inner selection median epoch = 5.
            </p>

            <div className="table-responsive">
              <table className="sweep-table font-mono">
                <thead>
                  <tr>
                    <th>Outer Fold</th>
                    <th>Accuracy</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>ROC-AUC</th>
                    <th>Sel. Epoch</th>
                  </tr>
                </thead>
                <tbody>
                  {foldRows.map((r) => (
                    <tr key={r.fold}>
                      <td>
                        <strong>{r.fold}</strong>
                      </td>
                      <td>{r.acc}</td>
                      <td style={{ color: 'var(--diag-uf-bar)' }}>{r.sens}</td>
                      <td style={{ color: 'var(--diag-nuf-bar)' }}>{r.spec}</td>
                      <td>{r.auc}</td>
                      <td>Ep {r.ep}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '16px', padding: '12px', background: 'var(--surface-sunken)', borderRadius: '6px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              <strong>Clinical Independence Note:</strong> Dataset lacks genuine hospital patient MRNs. Study-level grouping was enforced using acquisition metadata, but independent patient-level sampling cannot be guaranteed without hospital records.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
