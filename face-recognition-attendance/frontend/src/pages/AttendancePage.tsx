import { useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWebcam } from '../hooks/useWebcam';
import { verifyFace, type VerifyFaceResult } from '../services/studentService';
import './AttendancePage.css';

const AttendancePage = () => {
  const navigate = useNavigate();
  const { webcamActive, videoRef, startWebcam } = useWebcam();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerifyFaceResult | null>(null);

  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.8);
  }, [videoRef]);

  const handleVerify = useCallback(async () => {
    const imageData = captureFrame();
    if (!imageData) return;

    setIsVerifying(true);
    setResult(null);

    try {
      const res = await verifyFace(imageData);
      setResult(res);
    } catch (error) {
      console.error('Verification failed:', error);
      setResult({ matched: false, reason: 'Verification request failed. Please try again.' });
    } finally {
      setIsVerifying(false);
    }
  }, [captureFrame]);

  const handleRetry = () => {
    setResult(null);
  };

  return (
    <div className="attendance-page">
      <h1 className="attendance-page__title">Attendance Verification</h1>

      <main className="attendance-page__main">
        <div className="attendance-page__content">
          {/* Webcam Section */}
          <div className="attendance-page__webcam-section">
            <video ref={videoRef} autoPlay playsInline muted style={{ display: webcamActive ? 'block' : 'none' }} />
            <canvas ref={canvasRef} className="attendance-page__canvas" />

            {!webcamActive && (
              <button className="attendance-page__open-camera-btn" onClick={startWebcam}>
                Open Camera
              </button>
            )}

            {webcamActive && !isVerifying && !result && (
              <button className="attendance-page__verify-btn" onClick={handleVerify}>
                Verify Attendance
              </button>
            )}
          </div>

          {/* Result Section */}
          <div className="attendance-page__result-section">
            {!webcamActive && !result && (
              <p className="attendance-page__waiting">Open the camera to begin verification</p>
            )}

            {webcamActive && !isVerifying && !result && (
              <p className="attendance-page__waiting">Click "Verify Attendance" to check your face</p>
            )}

            {isVerifying && (
              <div className="attendance-page__verifying">
                <div className="attendance-page__spinner" />
                <p>Verifying face...</p>
              </div>
            )}

            {result && result.matched && (
              <div className="attendance-page__result-card attendance-page__result-card--present">
                <span className="attendance-page__result-icon">✅</span>
                <h2 className="attendance-page__result-status">Present</h2>
                <p className="attendance-page__result-detail">
                  Student ID: {result.student_id}
                </p>
                <p className="attendance-page__result-similarity">
                  Confidence: {((result.similarity ?? 0) * 100).toFixed(1)}%
                </p>
                <button className="attendance-page__retry-btn" onClick={handleRetry}>
                  Verify Again
                </button>
              </div>
            )}

            {result && !result.matched && (
              <div className="attendance-page__result-card attendance-page__result-card--absent">
                <span className="attendance-page__result-icon">❌</span>
                <h2 className="attendance-page__result-status">Absent</h2>
                <p className="attendance-page__result-detail">
                  {result.reason || 'Face did not match any registered student'}
                </p>
                {result.similarity !== undefined && (
                  <p className="attendance-page__result-similarity">
                    Best similarity: {(result.similarity * 100).toFixed(1)}%
                  </p>
                )}
                <button className="attendance-page__retry-btn" onClick={handleRetry}>
                  Try Again
                </button>
              </div>
            )}
          </div>
        </div>
      </main>

      <button className="attendance-page__back-btn" onClick={() => navigate('/')}>
        ← Back to Dashboard
      </button>
    </div>
  );
};

export default AttendancePage;