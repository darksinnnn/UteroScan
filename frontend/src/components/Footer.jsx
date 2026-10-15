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
            UteroScan is developed strictly for computational oncology research, algorithmic evaluation, and medical imaging education.
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
              Two-stage deep residual neural network for calibrated uterine fibroid detection from 2D pelvic B-mode ultrasound imaging.
            </p>
            <div className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)', marginTop: '14px' }}>
              REPRODUCIBLE RESEARCH BENCHMARK • 2026
            </div>
          </div>

          <div className="footer-links-col">
            <div className="footer-col-title font-mono">SPECIFICATIONS</div>
            <ul className="footer-spec-list font-mono">
              <li>Architecture: ResNet50 (Fine-Tuned)</li>
              <li>Input Resolution: 224 × 224 RGB</li>
              <li>Preprocessing: 3×3 Median + CLAHE 1.2</li>
              <li>Inference Latency: &lt; 20 ms / scan</li>
              <li>Training Schedule: Two-Stage (Head → Layer4)</li>
            </ul>
          </div>

          <div className="footer-links-col">
            <div className="footer-col-title font-mono">EVALUATION METRICS</div>
            <ul className="footer-spec-list font-mono">
              <li>Held-Out Test Set: 321 Clean Scans</li>
              <li>ROC-AUC: 0.9533 (PR-AUC: 0.9514)</li>
              <li>UF Sensitivity: 95.09% (τ=0.50)</li>
              <li>Optimal Accuracy: 89.41% (τ=0.65)</li>
              <li>Statistical Test: McNemar p = 0.0072</li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom-bar">
          <div style={{ fontSize: '0.82rem', color: 'var(--text-tertiary)' }}>
            © {new Date().getFullYear()} UteroScan Project. All rights reserved. Built in adherence with PRD.md & Frontend.md.
          </div>
          <div className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            DATASET: MENDELEY UF • 1,545 CLEAN B-MODE SCANS
          </div>
        </div>
      </div>
    </footer>
  );
}
