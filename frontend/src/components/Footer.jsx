import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container">
        {/* Prominent Academic Disclaimer Banner */}
        <div className="disclaimer-banner" style={{ marginBottom: '48px' }}>
          <ShieldAlert size={22} className="disclaimer-icon" />
          <div className="disclaimer-text">
            <strong>Investigational Academic Proof-of-Concept (Non-Clinical Disclaimer):</strong>
            <br />
            UteroScan is developed strictly for computational medical imaging research, algorithmic evaluation, and education.
            It is <strong>not</strong> an FDA-cleared, CE-marked, or certified medical diagnostic software device.
            This tool must never be employed as an independent clinical decision support system or substitute for expert sonographic interpretation by board-certified radiologists and gynecologists.
          </div>
        </div>

        {/* Footer Main Links & Technical Spec Grid */}
        <div className="footer-grid">
          <div className="footer-brand-col">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
              <div className="brand-mark" style={{ width: '28px', height: '28px', fontSize: '0.95rem' }}>U</div>
              <span className="brand-name" style={{ fontSize: '1.25rem' }}>UteroScan</span>
            </div>
            <p style={{ fontSize: '0.9rem', maxWidth: '340px', color: 'var(--text-secondary)' }}>
              Full-frame deep residual neural network (ResNet50 E0) for calibrated uterine fibroid detection from 2D pelvic B-mode ultrasound imaging.
            </p>
            <div className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)', marginTop: '14px' }}>
              REPRODUCIBLE RESEARCH BENCHMARK • 2026
            </div>
          </div>

          <div className="footer-links-col">
            <div className="footer-col-title font-mono">SPECIFICATIONS</div>
            <ul className="footer-spec-list font-mono">
              <li>Architecture: ResNet50 (Full-Frame E0)</li>
              <li>Input Resolution: 224 × 224 RGB</li>
              <li>Preprocessing: 3×3 Median + CLAHE 1.2</li>
              <li>Inference Latency: &lt; 20 ms / scan (GPU)</li>
              <li>Optimizer: Uniform Adam (lr=1e-4, wd=1e-4)</li>
            </ul>
          </div>

          <div className="footer-links-col">
            <div className="footer-col-title font-mono">EVALUATION METRICS</div>
            <ul className="footer-spec-list font-mono">
              <li>Development Cohort: 1,474 Eligible Scans</li>
              <li>Mean ROC-AUC: 0.9843 (PR-AUC: 0.9822)</li>
              <li>Mean Sensitivity: 94.75% (τ=0.50)</li>
              <li>Mean Outer Accuracy: 94.12% ± 0.64%</li>
              <li>Acquisition Proxy Groups: 917 Groups</li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom-bar">
          <div style={{ fontSize: '0.82rem', color: 'var(--text-tertiary)' }}>
            © {new Date().getFullYear()} UteroScan Project. All rights reserved. Authoritative E0 Production Baseline.
          </div>
          <div className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            DATASET: 1,474 RECOVERED DEV SCANS • 456 HELD-OUT EVALUATION SCANS
          </div>
        </div>
      </div>
    </footer>
  );
}
