import './StudentPage.css';
import { useWebcam } from '../hooks/useWebcam';
import { useFrameRecorder } from '../hooks/useFrameRecorder';
import WebcamView from '../components/student/WebcamView';
import FramePreview from '../components/student/FramePreview';
import SaveModal from '../components/student/SaveModal';
import StudentStatus from '../components/student/StudentStatus';

const StudentPage = () => {
  // Get student ID from localStorage (assuming it's stored after login)
  // const studentId = localStorage.getItem('studentId') || 'default-student-id';
  const studentId = "301481867"

  const { webcamActive, videoRef, startWebcam } = useWebcam();
  const {
    isRecording,
    capturedFrames,
    previewMode,
    canvasRef,
    startRecording,
    cancelRecording,
    confirmFrames,
    isSaving,
    message,
  } = useFrameRecorder(videoRef, studentId);

  return (
    <div className="student-page">
      <h1 className="student-page__title">Student Attendance</h1>
      <main className="student-page__main">
        <div className="student-page__top">
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
            onRetake={cancelRecording}
            isSaving={isSaving}
          />
          <StudentStatus
            studentName="John Doe"
            studentId={studentId}
            registered={false}
            registeredFacesCount={0}
          />
        </div>
        <div className="student-page__bottom">
          <FramePreview
            frames={capturedFrames}
          />
        </div>
      </main>
      <SaveModal isOpen={isSaving || message !== ''} message={message} />
    </div>
  );
};

export default StudentPage;