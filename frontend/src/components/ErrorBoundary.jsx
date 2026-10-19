import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            backgroundColor: '#F7F4EE',
            fontFamily: "'Plus Jakarta Sans', sans-serif",
            color: '#1F1C18',
          }}
        >
          <div
            style={{
              maxWidth: '560px',
              width: '100%',
              backgroundColor: '#FAF8F4',
              borderRadius: '16px',
              border: '1px solid #E5DDCF',
              boxShadow: '0 4px 20px rgba(33, 30, 26, 0.06)',
              padding: '36px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: 'rgba(184, 80, 50, 0.1)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '20px',
              }}
            >
              <AlertTriangle size={28} color="#B85032" />
            </div>

            <h2
              style={{
                fontSize: '1.4rem',
                fontWeight: 600,
                marginBottom: '10px',
                color: '#1F1C18',
              }}
            >
              Application Render Error
            </h2>

            <p
              style={{
                fontSize: '0.94rem',
                color: '#5D564C',
                lineHeight: 1.6,
                marginBottom: '20px',
              }}
            >
              An unexpected error occurred while rendering the clinical interface.
            </p>

            {this.state.error?.message && (
              <pre
                style={{
                  textAlign: 'left',
                  backgroundColor: '#EFE9DE',
                  padding: '14px',
                  borderRadius: '8px',
                  fontSize: '0.82rem',
                  fontFamily: "'JetBrains Mono', monospace",
                  color: '#8E2325',
                  overflowX: 'auto',
                  marginBottom: '24px',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {this.state.error.message}
              </pre>
            )}

            <button
              onClick={this.handleReload}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: '#B85032',
                color: '#FAF8F4',
                border: 'none',
                borderRadius: '8px',
                padding: '10px 20px',
                fontSize: '0.9rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              <RefreshCw size={16} /> Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
