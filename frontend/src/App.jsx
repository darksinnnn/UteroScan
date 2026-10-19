import React, { useEffect } from 'react';
import Lenis from 'lenis';
import { ArrowRight } from 'lucide-react';

import Navbar from './components/Navbar';
import ScanWaveVisual from './components/ScanWaveVisual';
import DiagnosticWorkspace from './components/DiagnosticWorkspace';
import PipelineExplainer from './components/PipelineExplainer';
import ClinicalAnalytics from './components/ClinicalAnalytics';
import MethodologyStory from './components/MethodologyStory';
import Footer from './components/Footer';
import './App.css';

export default function App() {
  // Initialize Lenis smooth scroll per Frontend.md Section 3
  useEffect(() => {
    // Respect prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const lenis = new Lenis({
      lerp: 0.1,
      duration: 1.2,
      smoothWheel: true,
      wheelMultiplier: 1.0,
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    const rafId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(rafId);
      lenis.destroy();
    };
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="uteroscan-app">
      <Navbar />

      {/* Hero Section */}
      <section id="hero" className="hero-section">
        <div className="container">
          <div className="hero-grid">
            {/* Left Column: Positioning & CTAs */}
            <div className="hero-content-col">
              <div className="hero-eyebrow-row">
                <span className="badge badge-neutral">GYNECOLOGIC ONCOLOGY AI</span>
                <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
                  PEER-REVIEWED METHODOLOGY
                </span>
              </div>

              <h1 className="hero-title">
                AI-assisted <em>uterine fibroid</em> detection from ultrasound.
              </h1>

              <p className="hero-subtitle">
                A calibrated, full-frame deep residual neural network (ResNet50 E0) designed for rapid, reproducible
                identification of uterine fibroids (UF) versus non-fibroid myometrium (NUF) from 2D pelvic B-mode sonography.
              </p>

              <div className="hero-cta-group">
                <button className="btn btn-primary" onClick={() => scrollTo('demo')}>
                  Launch Interactive Demo <ArrowRight size={16} />
                </button>
                <button className="btn btn-secondary" onClick={() => scrollTo('evidence')}>
                  Explore Empirical Evidence
                </button>
              </div>

              {/* Stats Ribbon */}
              <div className="hero-stats-ribbon">
                <div className="hero-stat-item">
                  <div className="hero-stat-num">0.9843</div>
                  <div className="hero-stat-lbl">Mean ROC-AUC (Nested CV)</div>
                </div>
                <div className="hero-stat-item">
                  <div className="hero-stat-num">94.75%</div>
                  <div className="hero-stat-lbl">UF Sensitivity (τ=0.50)</div>
                </div>
                <div className="hero-stat-item">
                  <div className="hero-stat-num">94.12%</div>
                  <div className="hero-stat-lbl">Outer Accuracy (τ=0.65)</div>
                </div>
                <div className="hero-stat-item">
                  <div className="hero-stat-num">&lt;20 ms</div>
                  <div className="hero-stat-lbl">GPU Inference Latency</div>
                </div>
              </div>
            </div>

            {/* Right Column: Hero Abstract Scan Visual */}
            <div className="hero-visual-col">
              <div className="hero-visual-card">
                <ScanWaveVisual />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Diagnostic Workspace (#demo) */}
      <section id="demo" className="section section-surface">
        <div className="container">
          <DiagnosticWorkspace />
        </div>
      </section>

      {/* The Four-Stage Pipeline (#pipeline) */}
      <section id="pipeline" className="section">
        <PipelineExplainer />
      </section>

      {/* Clinical Evidence & Validation (#evidence) */}
      <section id="evidence" className="section section-surface">
        <ClinicalAnalytics />
      </section>

      {/* Engineering Rigor & Methodology Story (#methodology) */}
      <section id="methodology" className="section">
        <MethodologyStory />
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}
