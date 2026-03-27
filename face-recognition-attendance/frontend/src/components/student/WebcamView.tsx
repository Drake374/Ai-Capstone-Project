import React from 'react';
import './WebcamView.css';

interface WebcamViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  webcamActive: boolean;
  isRecording: boolean;
  previewMode: boolean;
  onOpenCamera: () => void;
  onStartRecording: () => void;
  onCancelRecording: () => void;
  onConfirm: () => void;
  onRetake: () => void;
  isSaving?: boolean;
}

const WebcamView: React.FC<WebcamViewProps> = ({
  videoRef,
  canvasRef,
  webcamActive,
  isRecording,
  previewMode,
  onOpenCamera,
  onStartRecording,
  onCancelRecording,
  onConfirm,
  onRetake,
  isSaving = false,
}) => (
  <div className="webcam-view">
    {!webcamActive ? (
      <div className="webcam-placeholder">
        <button className="open-camera-btn" onClick={onOpenCamera}>
          Open Camera
        </button>
      </div>
    ) : (
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} />
        
        <div className="video-buttons">
          {!isRecording && !previewMode && (
            <button className="webcam-button" onClick={onStartRecording}>
              Start 15s Recording
            </button>
          )}

          {isRecording && (
            <button className="webcam-button" onClick={onCancelRecording}>
              Cancel
            </button>
          )}

          {previewMode && (
            <div className="webcam-preview-buttons">
              <button onClick={onConfirm} className="accept-btn" disabled={isSaving}>
                Accept
              </button>
              <button onClick={onRetake} className="retake-btn" disabled={isSaving}>
                Retake
              </button>
            </div>
          )}
        </div>
      </div>
    )}
  </div>
);

export default WebcamView;