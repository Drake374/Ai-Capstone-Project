import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './StudentPage.css';
import { useWebcam } from '../hooks/useWebcam';
import { useFrameRecorder } from '../hooks/useFrameRecorder';
import WebcamView from '../components/student/WebcamView';
import FramePreview from '../components/student/FramePreview';
import SaveModal from '../components/student/SaveModal';
import StudentStatus from '../components/student/StudentStatus';
import { getStudentProfile } from '../services/studentService';

const StudentPage = () => {
  // Get student info from localStorage (set during login)
  const user = JSON.parse(localStorage.getItem('user') || '{}');
  const studentId = user.studentId || '';
  const studentName = user.name || 'Unknown';
  const navigate = useNavigate();

  // Profile state from API
  const [registered, setRegistered] = useState(user.registered || false);
  const [faceCount, setFaceCount] = useState(0);

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

  // Fetch latest profile from API
  useEffect(() => {
    if (user.email) {
      getStudentProfile(user.email).then((profile) => {
        if (profile.found) {
          setRegistered(profile.registered || false);
          setFaceCount(profile.face_count || 0);
        }
      }).catch(console.error);
    }
  }, [user.email, message]); // re-fetch after save completes (message changes)

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
            studentName={studentName}
            studentId={studentId}
            registered={registered}
            registeredFacesCount={faceCount}
          />
        </div>
        <div className="student-page__bottom">
          <FramePreview
            frames={capturedFrames}
          />
        </div>
      </main>
      <SaveModal isOpen={isSaving || message !== ''} message={message} />

      <button className="student-page__back-btn" onClick={() => navigate('/')}>
        ← Back to Dashboard
      </button>
    </div>
  );
};

export default StudentPage;