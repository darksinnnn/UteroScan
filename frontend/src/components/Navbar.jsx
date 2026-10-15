import React, { useState, useEffect } from 'react';
import { Activity, ArrowUpRight, CheckCircle2 } from 'lucide-react';

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);

    // Health check poll
    fetch('/api/health')
      .then((res) => {
        if (res.ok) setApiOnline(true);
      })
      .catch(() => {
        setApiOnline(false);
      });

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className={`site-nav ${isScrolled ? 'scrolled' : ''}`}>
      <div className="container nav-container">
        {/* Brand */}
        <a href="#hero" className="nav-brand" onClick={(e) => { e.preventDefault(); scrollTo('hero'); }}>
          <div className="brand-mark">U</div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span className="brand-name">UteroScan</span>
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>
              DEEP ULTRASOUND ONCOLOGY
            </span>
          </div>
          <span className="brand-badge">RESEARCH BUILD</span>
        </a>

        {/* Links */}
        <nav>
          <ul className="nav-links">
            <li>
              <a href="#demo" className="nav-link" onClick={(e) => { e.preventDefault(); scrollTo('demo'); }}>
                Live Demo
              </a>
            </li>
            <li>
              <a href="#pipeline" className="nav-link" onClick={(e) => { e.preventDefault(); scrollTo('pipeline'); }}>
                How It Works
              </a>
            </li>
            <li>
              <a href="#evidence" className="nav-link" onClick={(e) => { e.preventDefault(); scrollTo('evidence'); }}>
                Clinical Evidence
              </a>
            </li>
            <li>
              <a href="#methodology" className="nav-link" onClick={(e) => { e.preventDefault(); scrollTo('methodology'); }}>
                Methodology
              </a>
            </li>
          </ul>
        </nav>

        {/* Action Button & Backend Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            className="api-status-indicator"
            title={apiOnline ? 'GPU Inference Server Connected' : 'Connecting to local API...'}
          >
            <span className={`status-dot ${apiOnline ? 'dot-online' : 'dot-pending'}`} />
            <span className="font-mono" style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
              {apiOnline ? 'ResNet50 Online' : 'Connecting...'}
            </span>
          </div>

          <button
            className="btn btn-primary"
            style={{ padding: '8px 18px', fontSize: '0.86rem' }}
            onClick={() => scrollTo('demo')}
          >
            Launch Test Scan
          </button>
        </div>
      </div>
    </header>
  );
}
