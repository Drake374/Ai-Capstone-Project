import './StudentPage.css';
import { useWebcam } from '../hooks/useWebcam';
import { useFrameRecorder } from '../hooks/useFrameRecorder';
import WebcamView from '../components/student/WebcamView';
import FramePreview from '../components/student/FramePreview';

const StudentPage = () => {
  const { webcamActive, videoRef, startWebcam } = useWebcam();
  const {
    isRecording,
    capturedFrames,
    previewMode,
    canvasRef,
    startRecording,
    cancelRecording,
    confirmFrames,
  } = useFrameRecorder(videoRef);

  return (
    <div className="student-page">
      <h1 className="student-page__title">Student Attendance</h1>
      <main className="student-page__main">
        <WebcamView
          videoRef={videoRef}
          canvasRef={canvasRef}
          webcamActive={webcamActive}
          isRecording={isRecording}
          previewMode={previewMode}
          onOpenCamera={startWebcam}
          onStartRecording={startRecording}
          onCancelRecording={cancelRecording}
          onConfirm={confirmFrames}
        />
        <FramePreview
          frames={capturedFrames}
          previewMode={previewMode}
          onConfirm={confirmFrames}
          onRetake={cancelRecording}
        />
      </main>
    </div>
  );
};

export default StudentPage;