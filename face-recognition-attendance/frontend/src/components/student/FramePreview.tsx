import React from 'react';
import type { CapturedFrame } from '../../types/frame';

interface FramePreviewProps {
  frames: CapturedFrame[];
  previewMode: boolean;
  onConfirm: () => void;
  onRetake: () => void;
}

// Buttons row height - kept constant so the strip height never shifts
const BUTTONS_H = 40;
const BUTTONS_MT = 8;

const FramePreview: React.FC<FramePreviewProps> = ({
  frames,
  previewMode,
  onConfirm,
  onRetake,
}) => (
  <div style={{
    flexShrink: 0,
    width: '100%',
    height: 'var(--preview-h)',
    display: 'flex',
    flexDirection: 'column',
    padding: '8px 8px',
    boxSizing: 'border-box',
  }}>

    {/* Horizontal scrolling strip */}
    <div style={{
      flex: 1,
      minHeight: 0,
      overflowX: 'auto',
      overflowY: 'hidden',
      display: 'flex',
      flexDirection: 'row',
      gap: '4px',
      alignItems: 'stretch',
      // Always reserve scroll track height so the strip height is stable
      scrollbarGutter: 'stable',
    }}>
      {frames.map((frame, i) => (
        <div
          key={frame.timestamp}
          style={{
            flexShrink: 0,
            // Width derived from height via aspect ratio so images are never tiny
            aspectRatio: '4 / 3',
            height: '100%',
            borderRadius: '4px',
            overflow: 'hidden',
            background: 'rgba(128,128,128,0.15)',
          }}
        >
          <img
            src={frame.imageData}
            alt={`Frame ${i + 1}`}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        </div>
      ))}

      {/* Empty placeholder slots so the strip is never blank before recording */}
      {frames.length === 0 && Array.from({ length: 8 }, (_, i) => (
        <div
          key={`placeholder-${i}`}
          style={{
            flexShrink: 0,
            aspectRatio: '4 / 3',
            height: '100%',
            borderRadius: '4px',
            background: 'rgba(128,128,128,0.1)',
          }}
        />
      ))}
    </div>

    {/* Buttons — visibility not conditional rendering, so height is always reserved */}
    <div style={{
      height: `${BUTTONS_H}px`,
      marginTop: `${BUTTONS_MT}px`,
      flexShrink: 0,
      display: 'flex',
      gap: '10px',
      visibility: previewMode ? 'visible' : 'hidden',
    }}>
      <button onClick={onConfirm} style={{ flex: 1 }}>Accept</button>
      <button onClick={onRetake}  style={{ flex: 1 }}>Retake</button>
    </div>

  </div>
);

export default FramePreview;