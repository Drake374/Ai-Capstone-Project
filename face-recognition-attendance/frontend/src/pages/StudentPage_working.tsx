import { useRef, useState, useCallback, useEffect } from 'react';

interface CapturedFrame {
  imageData: string;
  timestamp: number;
}

const StudentPage = () => {
  const [webcamActive, setWebcamActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<CapturedFrame[]>([]);
  const [previewMode, setPreviewMode] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recordingIntervalRef = useRef<number | null>(null);

  const startWebcam = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      setStream(mediaStream);
      setWebcamActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
    } catch (error) {
      console.error("Webcam error:", error);
    }
  }, []);

  const captureFrame = useCallback((): CapturedFrame | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return {
      imageData: canvas.toDataURL('image/jpeg', 0.8),
      timestamp: Date.now()
    };
  }, []);

  const startRecording = useCallback(() => {
    if (!webcamActive) return;

    setCapturedFrames([]);
    setIsRecording(true);
    setPreviewMode(false);

    let secondsElapsed = 0;
    const frames: CapturedFrame[] = [];

    const captureInterval = window.setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        frames.push(frame);
        setCapturedFrames([...frames]);
      }

      secondsElapsed++;
      if (secondsElapsed >= 15) {
        clearInterval(captureInterval);
        setIsRecording(false);
        setPreviewMode(true);
      }
    }, 1000);

    recordingIntervalRef.current = captureInterval;
  }, [webcamActive, captureFrame]);

  const cancelRecording = useCallback(() => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    setIsRecording(false);
    setCapturedFrames([]);
    setPreviewMode(false);
  }, []);

  const confirmFrames = useCallback(() => {
    const vectorizeAndSave = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/student/register-faces', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            frames: capturedFrames,
            studentId: 'current-student-id'
          })
        });

        if (response.ok) {
          setCapturedFrames([]);
          setPreviewMode(false);
        }
      } catch (error) {
        console.error('Save failed:', error);
      }
    };

    vectorizeAndSave();
  }, [capturedFrames]);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(e => console.error('Play failed:', e));
      };
    }
  }, [stream]);

  useEffect(() => {
    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  return (
    <div className="student-page" style={{ padding: '20px' }}>
      <h1>Student Attendance</h1>
      
      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        {!webcamActive ? (
          <button onClick={startWebcam} className="start-camera-btn">
            Open Camera
          </button>
        ) : (
          <>
            <div style={{ position: 'relative', width: '640px', height: '480px' }}>
              <video 
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              
              {!isRecording && !previewMode && (
                <button 
                  onClick={startRecording} 
                  style={{
                    position: 'absolute',
                    bottom: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '10px 20px'
                  }}
                >
                  Start 15s Recording
                </button>
              )}

              {isRecording && (
                <button 
                  onClick={cancelRecording} 
                  style={{
                    position: 'absolute',
                    bottom: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '10px 20px'
                  }}
                >
                  Cancel
                </button>
              )}
            </div>

            {capturedFrames.length > 0 && (
              <div style={{ width: '640px', height: '480px', overflowY: 'auto' }}>
                <h3 style={{ margin: '0 0 10px 0' }}>Captured ({capturedFrames.length}/15)</h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '5px'
                }}>
                  {capturedFrames.map((frame, index) => (
                    <img 
                      key={frame.timestamp}
                      src={frame.imageData}
                      alt={`Frame ${index + 1}`}
                      style={{ width: '100%', aspectRatio: '4/3', objectFit: 'cover' }}
                    />
                  ))}
                </div>

                {previewMode && (
                  <div style={{ marginTop: '10px', display: 'flex', gap: '10px' }}>
                    <button onClick={confirmFrames} style={{ flex: 1, padding: '8px' }}>
                      Accept
                    </button>
                    <button onClick={cancelRecording} style={{ flex: 1, padding: '8px' }}>
                      Retake
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default StudentPage;