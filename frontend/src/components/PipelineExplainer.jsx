import React from 'react';

export default function PipelineExplainer() {
  const steps = [
    {
      step: '01',
      title: 'B-Mode Ultrasound Ingestion',
      subtitle: 'Matrix Sanitization & Spatial Resampling',
      desc: 'Pelvic ultrasound scans are verified to exclude visible fetal anatomy and spectral Doppler waveforms. The image is converted to grayscale luminance and resampled to 224×224 using area interpolation (cv2.INTER_AREA) to preserve textural gradients.',
      tag: 'DIMENSION: 224×224'
    },
    {
      step: '02',
      title: 'Despeckle & Adaptive Contrast (CLAHE)',
      subtitle: 'Speckle Suppression & Tissue Standardization',
      desc: 'Ultrasound speckle noise is suppressed with a 3×3 median filter. Contrast-Limited Adaptive Histogram Equalization (clipLimit=1.2, 8×8 grid) equalizes non-uniform machine acoustic gains while preventing artificial artifact amplification.',
      tag: 'FILTER: MEDIAN + CLAHE 1.2'
    },
    {
      step: '03',
      title: 'Full-Frame ResNet50 Inference',
      subtitle: 'Deep Residual Feature Learning',
      desc: 'An ImageNet V2 pretrained ResNet50 with uniform Adam optimization (lr=1e-4) extracts hierarchical echotextural features across 50 deep convolutional layers to distinguish encapsulated fibroid acoustic shadows from diffuse myometrium.',
      tag: 'BACKBONE: RESNET50 (E0 BASELINE)'
    },
    {
      step: '04',
      title: 'Threshold-Calibrated Diagnostic Call',
      subtitle: 'Dual Operating Point Gating',
      desc: 'Two-class softmax logits yield calibrated posterior probabilities P(UF) and P(NUF). Operators can toggle between standard screening sensitivity (τ=0.50, 94.75% sensitivity) and optimal balanced accuracy (τ=0.65, 94.12% accuracy, 93.57% specificity).',
      tag: 'CALIBRATION: τ=0.50 / τ=0.65'
    }
  ];

  return (
    <div className="pipeline-section">
      <div className="container">
        <div style={{ maxWidth: '720px', marginBottom: '50px' }}>
          <span className="badge badge-neutral" style={{ marginBottom: '8px' }}>
            System Architecture
          </span>
          <h2>The Four-Stage Diagnostic Pipeline</h2>
          <p style={{ marginTop: '8px' }}>
            From raw ultrasound matrix to calibrated diagnostic decision: every step of the processing chain is engineered to eliminate scanner variability.
          </p>
        </div>

        {/* 4 Cards Grid */}
        <div className="pipeline-cards-grid">
          {steps.map((st) => (
            <div key={st.step} className="card pipeline-card">
              <div className="pipeline-step-badge font-mono">{st.step}</div>
              <h3 className="pipeline-card-title">{st.title}</h3>
              <div className="pipeline-card-subtitle">{st.subtitle}</div>
              <p className="pipeline-card-desc">{st.desc}</p>
              <div className="pipeline-card-tag font-mono">{st.tag}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
