import React from 'react';
import {
  Search,
  Filter,
  Layers,
  Sparkles,
  GitBranch,
  ShieldCheck,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export default function MethodologyStory() {
  return (
    <div className="methodology-story-section">
      <div className="container">
        {/* Section Header */}
        <div style={{ maxWidth: '780px', marginBottom: '48px' }}>
          <span className="badge badge-neutral" style={{ marginBottom: '8px' }}>
            Engineering & Methodological Rigor
          </span>
          <h2>A Diagnostic AI Built on Verifiable Clinical Evidence</h2>
          <p style={{ marginTop: '10px' }}>
            UteroScan pairs clinical image transformation with a two-stage deep convolutional architecture.
            Every claim is validated by empirical sweeps and paired statistical testing.
          </p>
        </div>

        {/* 4 Pillars Grid */}
        <div className="story-pillars-grid">
          {/* Pillar 1 */}
          <div className="card story-card">
            <div className="story-num font-mono">01</div>
            <h3 className="story-heading">Clinical Scope & Class Definition</h3>
            <p className="story-body">
              The model evaluates 2D pelvic B-mode ultrasound scans for focal uterine leiomyomas (UF).
              The negative class (NUF) represents homogeneous, physiological normal myometrium and non-fibroid
              tissue. All obstetric, gestational, and pulsed-wave Doppler modes are explicitly excluded.
            </p>
            <div className="story-footer-note font-mono">
              TARGET: FOCAL LEIOMYOMA MORPHOLOGY
            </div>
          </div>

          {/* Pillar 2 */}
          <div className="card story-card">
            <div className="story-num font-mono">02</div>
            <h3 className="story-heading">Phase 1 Clinical Transform Pipeline</h3>
            <p className="story-body">
              Raw ultrasound images suffer from machine-specific acoustic speckle and gain variations.
              Every input passes through <strong>cv2.INTER_AREA</strong> downsampling to 224×224, a <strong>3×3 median filter</strong> for
              despeckling, and <strong>CLAHE (clipLimit=1.2, 8×8 grid)</strong> to standardize acoustic tissue contrast before inference.
            </p>
            <div className="story-footer-note font-mono">
              STANDARDIZED ACOUSTIC CONTRAST
            </div>
          </div>

          {/* Pillar 3 */}
          <div className="card story-card">
            <div className="story-num font-mono">03</div>
            <h3 className="story-heading">Two-Stage Backbone Fine-Tuning</h3>
            <p className="story-body">
              Using ImageNet-pretrained ResNet50, training proceeds in two calibrated phases:
              <strong>Stage 1 (15 epochs)</strong> trains the classification head with frozen backbone at lr=1e-4;
              <strong>Stage 2 (10 epochs)</strong> unfreezes <code>layer4</code> (~30 deep convolutional layers) at lr=1e-5 to learn subtle echotextural margins.
            </p>
            <div className="story-footer-note font-mono">
              PRESERVES GLOBAL WEIGHT STABILITY
            </div>
          </div>

          {/* Pillar 4 */}
          <div className="card story-card">
            <div className="story-num font-mono">04</div>
            <h3 className="story-heading">Contamination Sweep & Robustness Check</h3>
            <p className="story-body">
              A comprehensive GPU EasyOCR audit discovered 202 unpunctuated obstetric scans (e.g. <code>GA 11w3d</code>)
              and spectral Doppler traces. Crucially, <strong>6 of the original 41 false positives were test-set contamination</strong>.
              On pure clean test scans, accuracy jumped from <strong>86.03% → 86.60%</strong>. Paired McNemar testing ($p=0.0072$)
              demonstrated the model's learned representations are highly robust against domain noise.
            </p>
            <div className="story-footer-note font-mono">
              MCNEMAR P = 0.0072 • ROBUST BASELINE
            </div>
          </div>
        </div>

        {/* Callout Box: What UteroScan Actually Classifies */}
        <div className="methodology-callout">
          <div className="callout-icon-col">
            <AlertCircle size={28} color="var(--accent-primary)" />
          </div>
          <div className="callout-content-col">
            <h4 style={{ fontSize: '1.05rem', fontWeight: '600', marginBottom: '4px' }}>
              Academic & Diagnostic Transparency Note (PRD Section 10)
            </h4>
            <p style={{ fontSize: '0.92rem', lineHeight: '1.55', color: 'var(--text-secondary)' }}>
              UteroScan is designed as an investigative AI screening aid for gynecological sonography, not an autonomous diagnostic device.
              Binary output distinguishes focal myometrial masses from physiological pelvic tissue. In clinical deployment, all AI findings must be interpreted
              in conjunction with patient history, transvaginal Doppler color flow mapping, and radiologist review.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
