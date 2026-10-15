import React, { useRef, useEffect } from 'react';

/**
 * ScanWaveVisual.jsx
 * Abstract acoustic wave propagation & ultrasound sector scan sweep.
 * Visual metaphor: "speckle noise -> coherent diagnostic clarity"
 * Built with vanilla HTML5 Canvas for instant rendering, zero network latency,
 * and automatic 60fps performance with mouse interaction.
 */
export default function ScanWaveVisual({ className = '' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let width = (canvas.width = canvas.parentElement.clientWidth || 500);
    let height = (canvas.height = canvas.parentElement.clientHeight || 460);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    // Mouse tracking for subtle acoustic refraction
    let mouse = { x: width * 0.5, y: height * 0.55, active: false };
    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      mouse.active = true;
    };
    const onMouseLeave = () => {
      mouse.active = false;
    };
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseleave', onMouseLeave);

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Simulation parameters
    let time = 0;
    const numArcs = 24;
    const numRayBeams = 32;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Center apex representing the ultrasound transducer head
      const apexX = width * 0.5;
      const apexY = 40;
      const maxRadius = height * 0.88;
      const sectorAngle = Math.PI * 0.55; // 100-degree ultrasound sector angle

      // Soft ambient background glow in warm sand/linen
      const bgGrad = ctx.createRadialGradient(apexX, apexY, 20, apexX, apexY, maxRadius);
      bgGrad.addColorStop(0, 'rgba(239, 233, 222, 0.45)');
      bgGrad.addColorStop(0.6, 'rgba(247, 244, 238, 0.2)');
      bgGrad.addColorStop(1, 'rgba(247, 244, 238, 0)');
      ctx.fillStyle = bgGrad;
      ctx.beginPath();
      ctx.moveTo(apexX, apexY);
      ctx.arc(apexX, apexY, maxRadius, Math.PI / 2 - sectorAngle / 2, Math.PI / 2 + sectorAngle / 2);
      ctx.closePath();
      ctx.fill();

      // Sector frame outline (hairline terracotta / warm charcoal)
      ctx.strokeStyle = 'rgba(184, 80, 50, 0.18)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw acoustic sector grid rays
      const startAngle = Math.PI / 2 - sectorAngle / 2;
      for (let i = 0; i <= numRayBeams; i++) {
        const rayAngle = startAngle + (sectorAngle * i) / numRayBeams;
        ctx.beginPath();
        ctx.moveTo(apexX, apexY);
        const rx = apexX + Math.cos(rayAngle) * maxRadius;
        const ry = apexY + Math.sin(rayAngle) * maxRadius;
        ctx.lineTo(rx, ry);
        ctx.strokeStyle = i % 8 === 0 ? 'rgba(93, 86, 76, 0.15)' : 'rgba(93, 86, 76, 0.05)';
        ctx.lineWidth = i % 8 === 0 ? 1 : 0.5;
        ctx.stroke();
      }

      // Draw concentric acoustic wave arcs propagating outward
      for (let j = 1; j <= numArcs; j++) {
        const baseRadius = (maxRadius * j) / numArcs;
        // Animated pulse
        const waveOffset = !prefersReducedMotion ? (time * 28) % (maxRadius / numArcs) : 0;
        const currentR = baseRadius + waveOffset;
        if (currentR > maxRadius) continue;

        // Influence of mouse position (acoustic tissue refraction)
        ctx.beginPath();
        const steps = 40;
        for (let s = 0; s <= steps; s++) {
          const a = startAngle + (sectorAngle * s) / steps;
          let r = currentR;

          // Local displacement
          const px = apexX + Math.cos(a) * r;
          const py = apexY + Math.sin(a) * r;
          const distToMouse = Math.hypot(px - mouse.x, py - mouse.y);
          if (distToMouse < 100) {
            r += Math.sin(distToMouse * 0.1 - time * 3) * (1 - distToMouse / 100) * 8;
          }

          const x = apexX + Math.cos(a) * r;
          const y = apexY + Math.sin(a) * r;
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }

        // Intensity increases at mid-depth (focal zone)
        const depthRatio = currentR / maxRadius;
        const focusFactor = Math.sin(depthRatio * Math.PI);
        const arcAlpha = 0.08 + focusFactor * 0.18;

        ctx.strokeStyle = `rgba(184, 80, 50, ${arcAlpha})`;
        ctx.lineWidth = depthRatio > 0.4 && depthRatio < 0.7 ? 1.5 : 0.75;
        ctx.stroke();
      }

      // Draw focal lesion simulation (a subtle, soft organoid mass in focal zone)
      const focalX = apexX + Math.sin(time * 0.5) * 15;
      const focalY = apexY + maxRadius * 0.55;
      const focalGrad = ctx.createRadialGradient(focalX, focalY, 4, focalX, focalY, 42);
      focalGrad.addColorStop(0, 'rgba(184, 80, 50, 0.22)');
      focalGrad.addColorStop(0.5, 'rgba(184, 80, 50, 0.08)');
      focalGrad.addColorStop(1, 'rgba(184, 80, 50, 0)');
      ctx.fillStyle = focalGrad;
      ctx.beginPath();
      ctx.arc(focalX, focalY, 42, 0, Math.PI * 2);
      ctx.fill();

      // Focal boundary markers (delicate clinical calipers)
      ctx.strokeStyle = 'rgba(184, 80, 50, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(focalX, focalY, 34, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Probe Transducer Element at top
      ctx.fillStyle = '#1F1C18';
      ctx.beginPath();
      ctx.arc(apexX, apexY, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#B85032';
      ctx.beginPath();
      ctx.arc(apexX, apexY, 5, 0, Math.PI * 2);
      ctx.fill();

      // Transducer label
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(93, 86, 76, 0.7)';
      ctx.textAlign = 'center';
      ctx.fillText('3.5 MHz CURVED ARRAY', apexX, apexY - 14);

      if (!prefersReducedMotion) {
        time += 0.02;
        animationFrameId = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseleave', onMouseLeave);
    };
  }, []);

  return (
    <div className={`scan-wave-wrapper ${className}`} style={{ position: 'relative', width: '100%', height: '100%', minHeight: '420px' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          cursor: 'crosshair',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: '14px',
          right: '18px',
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-tertiary)',
          pointerEvents: 'none',
          background: 'rgba(247, 244, 238, 0.85)',
          padding: '2px 8px',
          borderRadius: '4px',
          border: '1px solid var(--border-subtle)',
        }}
      >
        ACOUSTIC FIELD SIMULATION • 224×224 CLINI-NORM
      </div>
    </div>
  );
}
