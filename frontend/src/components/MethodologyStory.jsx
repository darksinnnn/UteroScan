import { AlertCircle } from 'lucide-react';

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
            UteroScan pairs clinical image transformation with a full-frame deep residual neural network.
            Every claim is validated by five-fold nested grouped cross-validation and rigorous data audits.
          </p>
        </div>

        {/* 4 Pillars Grid */}
        <div className="story-pillars-grid">
          {/* Pillar 1 */}
          <div className="card story-card">
            <div className="story-num font-mono">01</div>
            <h3 className="story-heading">Recovered Development Cohort</h3>
            <p className="story-body">
              Following an exhaustive image-level audit, <strong>1,474 eligible development scans</strong> (685 UF, 789 NUF)
              were recovered across <strong>917 acquisition-study proxy groups</strong>. 64 scans with visible fetal anatomy
              were quarantined, and <strong>456 scans</strong> were isolated into an untouched held-out evaluation pool.
            </p>
            <div className="story-footer-note font-mono">
              1,474 DEV SCANS • 917 STUDY GROUPS
            </div>
          </div>

          {/* Pillar 2 */}
          <div className="card story-card">
            <div className="story-num font-mono">02</div>
            <h3 className="story-heading">Standardized Clinical Transform</h3>
            <p className="story-body">
              Raw ultrasound scans suffer from machine-specific acoustic speckle and gain variations.
              Every input passes through <strong>cv2.INTER_AREA</strong> downsampling to 224×224, a <strong>3×3 median filter</strong> for
              despeckling, and <strong>CLAHE (clipLimit=1.2, 8×8 grid)</strong> to standardize tissue contrast before inference.
            </p>
            <div className="story-footer-note font-mono">
              STANDARDIZED ACOUSTIC CONTRAST
            </div>
          </div>

          {/* Pillar 3 */}
          <div className="card story-card">
            <div className="story-num font-mono">03</div>
            <h3 className="story-heading">ResNet50 E0 Baseline Architecture</h3>
            <p className="story-body">
              Utilizes torchvision ResNet50 with ImageNet V2 pretrained weights and a uniform Adam learning rate of
              <strong> 1e-4</strong> (weight decay 1e-4). Models are selected via <strong>5-fold nested grouped cross-validation</strong>,
              ensuring outer folds are never touched during checkpoint selection.
            </p>
            <div className="story-footer-note font-mono">
              UNIFORM ADAM LR 1E-4 • 5-FOLD NESTED CV
            </div>
          </div>

          {/* Pillar 4 */}
          <div className="card story-card">
            <div className="story-num font-mono">04</div>
            <h3 className="story-heading">Rigorous Controlled Ablations</h3>
            <p className="story-body">
              Systematic experiments evaluated <strong>E1 viewport cropping</strong>, <strong>E2 caliper augmentation</strong>,
              <strong> E3 auxiliary localization</strong>, and <strong>E5 differential learning rates</strong>. None consistently
              outperformed E0 in nested validation. E0 remains the authoritative development baseline with <strong>94.12% accuracy</strong> and <strong>0.9843 ROC-AUC</strong>.
            </p>
            <div className="story-footer-note font-mono">
              CONTROLLED E0–E5 TRAJECTORY • E0 LOCKED
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
