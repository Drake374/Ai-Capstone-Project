import React from 'react';
import type { CapturedFrame } from '../../types/frame';
import './FramePreview.css';

interface FramePreviewProps {
  frames: CapturedFrame[];
}

const FramePreview: React.FC<FramePreviewProps> = ({
  frames
}) => (
  <div className="frame-preview-container">
    <div className="frame-preview-scroll">
      {frames.map((frame, i) => (
        <div key={frame.timestamp} className="frame-item">
          <img
            src={frame.imageData}
            alt={`Frame ${i + 1}`}
          />
        </div>
      ))}

      {frames.length === 0 && Array.from({ length: 8 }, (_, i) => (
        <div key={`placeholder-${i}`} className="frame-placeholder" />
      ))}
    </div>
  </div>
);

export default FramePreview;