import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Upload,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sliders,
  FileImage,
  ShieldCheck
} from 'lucide-react';

export default function DiagnosticWorkspace() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // Pipeline execution states: 'idle' | 'uploading' | 'preprocessing' | 'inferring' | 'complete' | 'error'
  const [processState, setProcessState] = useState('idle');
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const [customThreshold, setCustomThreshold] = useState(0.65);
  const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
  const fileInputRef = useRef(null);

  const runInference = useCallback(async ({ sampleId, file }) => {
    setResult(null);
    setErrorMsg(null);
    setProcessState('uploading');

    try {
      if (file && file.size > 20 * 1024 * 1024) {
        throw new Error('Image file exceeds maximum allowable size of 20MB.');
      }
      const formData = new FormData();
      if (sampleId) {
        formData.append('sample_id', sampleId);
      } else if (file) {
        formData.append('file', file);
      }
      formData.append('custom_threshold', customThreshold.toString());

      // Simulate subtle stage transition for visual clarity
      setTimeout(() => setProcessState('preprocessing'), 180);
      setTimeout(() => setProcessState('inferring'), 360);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Inference service returned an error.');
      }

      const data = await res.json();
      setTimeout(() => {
        setResult(data);
        setProcessState('complete');
      }, 550);
    } catch (err) {
      console.error('Inference error:', err);
      setErrorMsg(err.message || 'Error processing ultrasound image.');
      setProcessState('error');
    }
  }, [API_BASE, customThreshold]);

  const handleSelectSample = useCallback((sample) => {
    setSelectedSample(sample);
    setImagePreview(sample.thumbnail_base64 || null);
    runInference({ sampleId: sample.id });
  }, [runInference]);

  // Fetch pre-curated samples on load
  useEffect(() => {
    fetch(`${API_BASE}/api/samples`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch samples');
        return res.json();
      })
      .then((data) => {
        setSamples(data);
        if (data && data.length > 0) {
          handleSelectSample(data[0]);
        }
      })
      .catch((err) => {
        console.warn('Backend samples API unavailable:', err);
        setErrorMsg('Backend inference API is unreachable. Please verify the backend is running (`npm run dev:backend`).');
        setProcessState('error');
      });
  }, [API_BASE, handleSelectSample]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate image file
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please select a valid ultrasound image file (JPG, PNG, TIFF).');
      return;
    }

    setSelectedSample(null);
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result);
    };
    reader.readAsDataURL(file);

    runInference({ file });
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedSample(null);
      const reader = new FileReader();
      reader.onload = (event) => {
        setImagePreview(event.target?.result);
      };
      reader.readAsDataURL(file);
      runInference({ file });
    }
  };


  // Determine current active diagnostic call based on threshold slider
  const probUF = result ? result.probabilities.UF : 0;
  const isPositiveAtCurrentThreshold = probUF >= customThreshold;

  return (
    <div className="diagnostic-workspace">
      {/* Top Bar: Sample Selector Pills */}
      <div className="workspace-header">
        <div>
          <span className="badge badge-neutral" style={{ marginBottom: '8px' }}>
            Interactive Diagnostic Workspace
          </span>
          <h2 style={{ fontSize: '1.9rem', marginBottom: '6px' }}>Evaluate Clinical Ultrasound Scans</h2>
          <p style={{ fontSize: '0.96rem' }}>
            Test anonymized held-out clinical validation cases or upload your own B-mode pelvic scans for immediate ResNet50 analysis.
          </p>
        </div>

        {/* Curated Sample Buttons */}
        <div className="samples-tray">
          <div className="samples-tray-label">Pre-Loaded Test Cohort:</div>
          <div className="samples-list">
            {samples.map((s) => {
              const isSelected = selectedSample?.id === s.id;
              const isUF = s.ground_truth === 'UF';
              return (
                <button
                  key={s.id}
                  onClick={() => handleSelectSample(s)}
                  className={`sample-pill ${isSelected ? 'active' : ''}`}
                >
                  <span className={`pill-dot ${isUF ? 'dot-uf' : 'dot-nuf'}`} />
                  <span className="pill-title">{s.title.split(':')[0]}</span>
                  <span className="pill-type">({s.ground_truth})</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main 2-Column Grid: Image Input/Preprocessing vs. Diagnostic Classification */}
      <div className="workspace-grid">
        {/* Left Column: Image Dropzone & Before/After Preprocessing */}
        <div className="workspace-column">
          <div className="card workspace-card">
            <div className="card-header-row">
              <div className="card-title-group">
                <FileImage size={18} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '1.15rem' }}>Ultrasound Input & Preprocessing</h3>
              </div>
              <button
                className="btn btn-outline"
                style={{ padding: '6px 12px', fontSize: '0.82rem' }}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={14} /> Upload Custom Scan
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </div>

            {/* Drop / Display Area */}
            <div
              className={`dropzone-area ${processState !== 'idle' && processState !== 'complete' ? 'processing' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              {imagePreview ? (
                <div className="viewer-split">
                  {/* Raw Input */}
                  <div className="viewer-pane">
                    <div className="pane-tag">Raw Acquisition (B-Mode)</div>
                    <img src={imagePreview} alt="Raw Ultrasound" className="scan-img" />
                    <div className="pane-caption">Original Ultrasound Matrix</div>
                  </div>

                  {/* Preprocessed (CLAHE + Median) */}
                  <div className="viewer-pane">
                    <div className="pane-tag pane-tag-accent">Phase 1 Preprocessed</div>
                    {result?.visualizations?.preprocessed_image ? (
                      <img
                        src={result.visualizations.preprocessed_image}
                        alt="Preprocessed CLAHE"
                        className="scan-img"
                      />
                    ) : (
                      <div className="scan-placeholder">
                        <RefreshCw size={24} className="spin-icon" />
                        <span>Applying CLAHE & Despeckle...</span>
                      </div>
                    )}
                    <div className="pane-caption">224×224 • CLAHE 1.2 • 3×3 Median</div>
                  </div>
                </div>
              ) : (
                <div className="empty-dropzone" onClick={() => fileInputRef.current?.click()}>
                  <Upload size={36} color="var(--accent-primary)" />
                  <div style={{ marginTop: '12px', fontWeight: '600', color: 'var(--text-primary)' }}>
                    Drop a Pelvic Ultrasound scan here, or click to upload
                  </div>
                  <div style={{ fontSize: '0.84rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    Supports anonymized JPG, PNG, and DICOM-exported formats
                  </div>
                </div>
              )}
            </div>

            {/* Pipeline Stage Indicator */}
            <div className="pipeline-stepper">
              <div className={`step-item ${processState !== 'idle' ? 'step-active' : ''}`}>
                <div className="step-circle">1</div>
                <div className="step-label">Ingestion</div>
              </div>
              <div className="step-line" />
              <div className={`step-item ${processState === 'preprocessing' || processState === 'inferring' || processState === 'complete' ? 'step-active' : ''}`}>
                <div className="step-circle">2</div>
                <div className="step-label">Median & CLAHE</div>
              </div>
              <div className="step-line" />
              <div className={`step-item ${processState === 'inferring' || processState === 'complete' ? 'step-active' : ''}`}>
                <div className="step-circle">3</div>
                <div className="step-label">ResNet50 Backbone</div>
              </div>
              <div className="step-line" />
              <div className={`step-item ${processState === 'complete' ? 'step-active' : ''}`}>
                <div className="step-circle">4</div>
                <div className="step-label">Diagnostic Output</div>
              </div>
            </div>

            {/* Latency Readout */}
            {result?.latency && (
              <div className="latency-bar">
                <span className="font-mono">
                  Preprocessing: <strong>{result.latency.preprocessing_ms} ms</strong>
                </span>
                <span className="separator">•</span>
                <span className="font-mono">
                  ResNet50 Inference: <strong>{result.latency.inference_ms} ms</strong>
                </span>
                <span className="separator">•</span>
                <span className="font-mono">
                  Total End-to-End: <strong>{result.latency.total_ms} ms</strong>
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Diagnostic Interpretation & Calibrated Confidence */}
        <div className="workspace-column">
          <div className="card workspace-card diagnostic-result-card">
            <div className="card-header-row">
              <div className="card-title-group">
                <ShieldCheck size={18} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '1.15rem' }}>Diagnostic Inference & Calibration</h3>
              </div>
              {result && (
                <span className={`badge ${isPositiveAtCurrentThreshold ? 'badge-uf' : 'badge-nuf'}`}>
                  {isPositiveAtCurrentThreshold ? 'UF Positivity' : 'Normal / Negative'}
                </span>
              )}
            </div>

            {errorMsg && (
              <div className="error-box">
                <AlertTriangle size={18} />
                <span>{errorMsg}</span>
              </div>
            )}

            {result ? (
              <div className="result-content-layout">
                {/* Headline Result Box */}
                <div className={`result-headline-banner ${isPositiveAtCurrentThreshold ? 'banner-uf' : 'banner-nuf'}`}>
                  <div className="headline-icon-col">
                    {isPositiveAtCurrentThreshold ? (
                      <AlertTriangle size={32} color="var(--diag-uf-text)" />
                    ) : (
                      <CheckCircle2 size={32} color="var(--diag-nuf-text)" />
                    )}
                  </div>
                  <div className="headline-text-col">
                    <div className="result-main-call">
                      {isPositiveAtCurrentThreshold ? 'Uterine Fibroid Detected' : 'Normal Myometrium / Non-Fibroid'}
                    </div>
                    <div className="result-sub-call">
                      {isPositiveAtCurrentThreshold
                        ? 'Focal acoustic disruption consistent with uterine leiomyoma morphology.'
                        : 'Homogeneous myometrial echogenicity without discrete encapsulated mass.'}
                    </div>
                  </div>
                </div>

                {/* Calibrated Confidence Probability Bar */}
                <div className="confidence-meter-group">
                  <div className="meter-label-row">
                    <span className="meter-name font-mono">P(UF | Fibroid Lesion):</span>
                    <span className="meter-val font-mono">{(probUF * 100).toFixed(1)}%</span>
                  </div>
                  <div className="meter-track">
                    <div
                      className="meter-fill fill-uf"
                      style={{ width: `${Math.max(4, Math.min(100, probUF * 100))}%` }}
                    />
                    {/* Operating Threshold Marker */}
                    <div
                      className="threshold-marker"
                      style={{ left: `${customThreshold * 100}%` }}
                      title={`Decision Cutoff: ${(customThreshold * 100).toFixed(0)}%`}
                    >
                      <span className="marker-pin" />
                      <span className="marker-tag">τ = {customThreshold.toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="meter-label-row" style={{ marginTop: '16px' }}>
                    <span className="meter-name font-mono">P(NUF | Normal Myometrium):</span>
                    <span className="meter-val font-mono">{((1 - probUF) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="meter-track">
                    <div
                      className="meter-fill fill-nuf"
                      style={{ width: `${Math.max(4, Math.min(100, (1 - probUF) * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Threshold Calibration Control Box */}
                <div className="threshold-tuner-card">
                  <div className="tuner-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sliders size={16} color="var(--text-primary)" />
                      <strong style={{ fontSize: '0.92rem' }}>Clinical Threshold Calibration</strong>
                    </div>
                    <div className="tuner-preset-pills">
                      <button
                        className={`preset-btn ${customThreshold === 0.50 ? 'active' : ''}`}
                        onClick={() => setCustomThreshold(0.50)}
                      >
                        τ = 0.50 (Standard)
                      </button>
                      <button
                        className={`preset-btn ${customThreshold === 0.65 ? 'active' : ''}`}
                        onClick={() => setCustomThreshold(0.65)}
                      >
                        τ = 0.65 (Optimal)
                      </button>
                    </div>
                  </div>

                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '8px 0 14px' }}>
                    Adjusting cutoff isolates clinical priorities: Standard <strong>τ = 0.50</strong> prioritizes
                    screening sensitivity (<strong>94.75%</strong>), while <strong>τ = 0.65</strong> optimizes
                    overall diagnostic accuracy to <strong>94.12%</strong> and specificity to <strong>93.57%</strong>.
                  </p>

                  <div className="slider-row">
                    <span className="slider-bound font-mono">0.30</span>
                    <input
                      type="range"
                      min="0.30"
                      max="0.80"
                      step="0.05"
                      value={customThreshold}
                      onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                      className="threshold-slider"
                    />
                    <span className="slider-bound font-mono">0.80</span>
                    <span className="slider-active-val font-mono">Cutoff: {customThreshold.toFixed(2)}</span>
                  </div>
                </div>

                {/* Clinical Context / Metadata Note */}
                <div className="case-meta-box">
                  <div className="meta-title font-mono">SPECIMEN & PIPELINE CONTEXT</div>
                  <div className="meta-grid">
                    <div>
                      <span className="meta-label">Selected Cohort:</span>
                      <span className="meta-val">{result.source}</span>
                    </div>
                    <div>
                      <span className="meta-label">Network Architecture:</span>
                      <span className="meta-val">ResNet50 (Full-Frame E0 Baseline)</span>
                    </div>
                    <div>
                      <span className="meta-label">Development Benchmark:</span>
                      <span className="meta-val">0.9843 ROC-AUC • 94.75% Sens • 94.12% Acc</span>
                    </div>
                    <div>
                      <span className="meta-label">Risk Category:</span>
                      <span className="meta-val">{result.classification.risk_level}</span>
                    </div>
                  </div>
                </div>

                {/* Research Prototype Disclaimer */}
                <div style={{ marginTop: '14px', padding: '10px 14px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  <strong>Research Prototype Disclaimer:</strong> Predictions are generated by an investigational deep neural network and do not constitute a clinical diagnosis. All sonograms must be evaluated by a certified medical specialist.
                </div>
              </div>
            ) : (
              <div className="awaiting-state">
                <RefreshCw
                  size={32}
                  className={processState !== 'idle' && processState !== 'error' ? 'spin-icon' : ''}
                  color="var(--border-strong)"
                />
                <div style={{ marginTop: '14px', fontWeight: '500', color: 'var(--text-secondary)' }}>
                  {processState === 'idle'
                    ? 'Select a test scan or upload an image to begin evaluation.'
                    : processState === 'error'
                    ? 'Inference engine disconnected. Ensure backend is running or upload a scan.'
                    : 'Running clinical preprocessing and deep neural inference...'}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
