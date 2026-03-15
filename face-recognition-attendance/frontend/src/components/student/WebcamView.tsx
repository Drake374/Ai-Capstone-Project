import React from 'react';

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
}) => (
  <div style={{
    flex: 1,
    minHeight: 0,
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  }}>
    {!webcamActive ? (
      <button onClick={onOpenCamera}>Open Camera</button>
    ) : (
      <>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            width: 'auto',
            height: '100%',
            aspectRatio: '4/3',
            objectFit: 'cover',
          }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {!isRecording && !previewMode && (
          <button style={{
            position: 'absolute',
            bottom: '16px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 20px',
          }}
            onClick={onStartRecording}
          >
            Start 15s Recording
          </button>
        )}

        {isRecording && (
          <button style={{
            position: 'absolute',
            bottom: '16px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 20px',
          }}
            onClick={onCancelRecording}
          >
            Cancel
          </button>
        )}

        {/* {!isRecording && previewMode && (
          <button style={{
            position: 'sticky',
            bottom: '16px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 20px',
          }}
            onClick={onCancelRecording}
          >
            Retake
          </button>
        )} */}
      </>
    )}
  </div>
);

export default WebcamView;