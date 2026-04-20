import { useRef, useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWebcam } from '../hooks/useWebcam';
import { verifyFace, type VerifyFaceResult } from '../services/studentService';
import './AttendancePage.css';

const SESSION_DURATION_MS = 45 * 60 * 1000;
const MIN_CHECK_DELAY_MS = 2 * 60 * 1000;
const MAX_CHECK_DELAY_MS = 8 * 60 * 1000;
const FINAL_WINDOW_MS = 8 * 60 * 1000;
const CHALLENGE_DELAY_MS = 3000;
const MIN_SUCCESSFUL_CHECKS = 4;
const MIN_SUCCESS_RATE = 0.7;

const CHALLENGES = [
  'Turn your face slightly left',
  'Turn your face slightly right',
  'Lean a little closer to the camera',
  'Sit upright and keep your full face visible',
];

type SessionOutcome = 'present' | 'partial' | 'absent';

const AttendancePage = () => {
  const navigate = useNavigate();
  const { webcamActive, videoRef, startWebcam } = useWebcam();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<number | null>(null);

  const user = JSON.parse(localStorage.getItem('user') || '{}');
  const expectedStudentId = user.studentId || '';

  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerifyFaceResult | null>(null);
  const [monitoringActive, setMonitoringActive] = useState(false);
  const [successfulChecks, setSuccessfulChecks] = useState(0);
  const [failedChecks, setFailedChecks] = useState(0);
  const [checkNumber, setCheckNumber] = useState(0);
  const [lateWindowCheckPassed, setLateWindowCheckPassed] = useState(false);
  const [sessionStartedAt, setSessionStartedAt] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionEndsAt, setSessionEndsAt] = useState<number | null>(null);
  const [sessionTimeRemainingLabel, setSessionTimeRemainingLabel] = useState('45:00');
  const [monitoringMessage, setMonitoringMessage] = useState(
    'Open the camera and start attendance monitoring. Random checks continue across the full class session.'
  );
  const [sessionComplete, setSessionComplete] = useState(false);
  const [sessionOutcome, setSessionOutcome] = useState<SessionOutcome | null>(null);
  const [activeChallenge, setActiveChallenge] = useState<string | null>(null);

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

  const clearAttendanceTimers = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const formatDuration = useCallback((milliseconds: number) => {
    const totalSeconds = Math.max(0, Math.ceil(milliseconds / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  const finalizeSession = useCallback((
    checksRun: number,
    checksPassed: number,
    lateCheckPassed: boolean,
  ) => {
    clearAttendanceTimers();
    setMonitoringActive(false);
    setIsVerifying(false);
    setActiveChallenge(null);
    setSessionComplete(true);
    setSessionEndsAt(null);
    setSessionTimeRemainingLabel('0:00');

    const successRate = checksRun > 0 ? checksPassed / checksRun : 0;
    const hasStrongCoverage =
      checksPassed >= MIN_SUCCESSFUL_CHECKS &&
      successRate >= MIN_SUCCESS_RATE &&
      lateCheckPassed;

    if (hasStrongCoverage) {
      setSessionOutcome('present');
      setMonitoringMessage(
        'Session complete. Attendance marked present because the student stayed verified across the session, including the late check window.'
      );
      return;
    }

    if (checksPassed > 0) {
      setSessionOutcome('partial');
      setMonitoringMessage(
        'Session complete. Attendance marked partial because the student was not verified strongly enough across the full session.'
      );
      return;
    }

    setSessionOutcome('absent');
    setMonitoringMessage(
      'Session complete. Attendance marked absent because the student was not verified during the session.'
    );
  }, [clearAttendanceTimers]);

  const runVerification = useCallback(async (label: string) => {
    const imageData = captureFrame();
    if (!imageData || !expectedStudentId || !sessionId) {
      setMonitoringMessage('Missing student identity, session ID, or camera frame for verification.');
      setMonitoringActive(false);
      clearAttendanceTimers();
      return;
    }

    setIsVerifying(true);
    setResult(null);
    setMonitoringMessage(`${label} in progress...`);

    try {
      const res = await verifyFace(imageData, sessionId, expectedStudentId);
      const now = Date.now();
      const inLateWindow = sessionEndsAt !== null && sessionEndsAt - now <= FINAL_WINDOW_MS;
      const nextCheckCount = checkNumber + 1;

      setResult(res);
      setCheckNumber(nextCheckCount);

      if (res.matched) {
        const nextSuccessfulChecks = successfulChecks + 1;
        setSuccessfulChecks(nextSuccessfulChecks);

        if (inLateWindow) {
          setLateWindowCheckPassed(true);
        }

        setMonitoringMessage(
          inLateWindow
            ? 'Late-session verification passed. Monitoring continues until the session ends.'
            : 'Verification passed. Attendance monitoring stays active for later random checks.'
        );
      } else {
        const nextFailedChecks = failedChecks + 1;
        setFailedChecks(nextFailedChecks);
        setMonitoringMessage(
          res.reason || 'Verification failed. Stay visible because more random checks will happen later.'
        );
      }
    } catch (error) {
      console.error('Verification failed:', error);
      setFailedChecks((current) => current + 1);
      setResult({ matched: false, reason: 'Verification request failed. Please stay ready for later checks.' });
      setMonitoringMessage('Verification request failed. Monitoring will continue with later random checks.');
    } finally {
      setActiveChallenge(null);
      setIsVerifying(false);
    }
  }, [
    captureFrame,
    checkNumber,
    clearAttendanceTimers,
    expectedStudentId,
    failedChecks,
    sessionEndsAt,
    successfulChecks,
    sessionId,
  ]);

  const scheduleNextCheck = useCallback(() => {
    if (!monitoringActive || !sessionEndsAt) {
      return;
    }

    const now = Date.now();
    const remaining = sessionEndsAt - now;
    if (remaining <= 0) {
      finalizeSession(checkNumber, successfulChecks, lateWindowCheckPassed);
      return;
    }

    let minDelay = MIN_CHECK_DELAY_MS;
    let maxDelay = MAX_CHECK_DELAY_MS;

    if (!lateWindowCheckPassed) {
      const latestBeforeFinalWindow = Math.max(15 * 1000, remaining - FINAL_WINDOW_MS);
      maxDelay = Math.min(maxDelay, latestBeforeFinalWindow);
    } else {
      maxDelay = Math.min(maxDelay, remaining - 15 * 1000);
    }

    if (remaining <= FINAL_WINDOW_MS) {
      minDelay = 30 * 1000;
      maxDelay = Math.min(2 * 60 * 1000, remaining - 15 * 1000);
    }

    if (maxDelay < minDelay) {
      minDelay = Math.max(15 * 1000, Math.min(minDelay, maxDelay));
    }

    const delayLowerBound = Math.max(15 * 1000, minDelay);
    const delayUpperBound = Math.max(delayLowerBound, maxDelay);
    const delay =
      Math.floor(Math.random() * (delayUpperBound - delayLowerBound + 1)) + delayLowerBound;

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }

    timerRef.current = window.setTimeout(() => {
      const challenge =
        Math.random() < 0.6
          ? CHALLENGES[Math.floor(Math.random() * CHALLENGES.length)]
          : null;

      if (challenge) {
        setActiveChallenge(challenge);
        setMonitoringMessage(
          'A live check is starting. Follow the on-screen instruction and stay in frame.'
        );
        window.setTimeout(() => {
          void runVerification('Random re-check');
        }, CHALLENGE_DELAY_MS);
        return;
      }

      void runVerification('Random re-check');
    }, delay);
  }, [
    checkNumber,
    finalizeSession,
    lateWindowCheckPassed,
    monitoringActive,
    runVerification,
    sessionEndsAt,
    successfulChecks,
  ]);

  useEffect(() => {
    if (!monitoringActive || !sessionEndsAt) {
      return;
    }

    const updateRemainingTime = () => {
      const remaining = sessionEndsAt - Date.now();
      setSessionTimeRemainingLabel(formatDuration(remaining));

      if (remaining <= 0) {
        finalizeSession(checkNumber, successfulChecks, lateWindowCheckPassed);
      }
    };

    updateRemainingTime();
    const interval = window.setInterval(updateRemainingTime, 1000);
    return () => window.clearInterval(interval);
  }, [
    checkNumber,
    finalizeSession,
    formatDuration,
    lateWindowCheckPassed,
    monitoringActive,
    sessionEndsAt,
    successfulChecks,
  ]);

  useEffect(() => {
    if (!monitoringActive || isVerifying || sessionComplete || activeChallenge || checkNumber === 0) {
      return;
    }

    scheduleNextCheck();
  }, [activeChallenge, checkNumber, isVerifying, monitoringActive, scheduleNextCheck, sessionComplete]);

  useEffect(() => {
    return () => {
      clearAttendanceTimers();
    };
  }, [clearAttendanceTimers]);

  const handleStartMonitoring = useCallback(async () => {
    if (!webcamActive) {
      await startWebcam();
    }

    const startTime = Date.now();
    const endTime = startTime + SESSION_DURATION_MS;

    clearAttendanceTimers();
    setMonitoringActive(true);
    setSessionComplete(false);
    setSessionOutcome(null);
    setSuccessfulChecks(0);
    setFailedChecks(0);
    setCheckNumber(0);
    setLateWindowCheckPassed(false);
    setActiveChallenge(null);
    setResult(null);
    const newSessionId = `session-${startTime}`;
    setSessionId(newSessionId);
    setSessionStartedAt(startTime);
    setSessionEndsAt(endTime);
    setSessionTimeRemainingLabel(formatDuration(SESSION_DURATION_MS));
    setMonitoringMessage(
      'Attendance monitoring started. Random re-checks will continue across the full session.'
    );

    await runVerification('Initial verification');
  }, [
    clearAttendanceTimers,
    finalizeSession,
    formatDuration,
    runVerification,
    startWebcam,
    webcamActive,
  ]);

  const handleStopMonitoring = useCallback(() => {
    clearAttendanceTimers();
    setMonitoringActive(false);
    setActiveChallenge(null);
    setSessionComplete(false);
    setSessionOutcome(null);
    setSessionId(null);
    setSessionEndsAt(null);
    setMonitoringMessage('Attendance monitoring stopped before the session completed.');
  }, [clearAttendanceTimers]);

  const handleDismissResult = () => {
    setResult(null);
  };

  const outcomeLabel =
    sessionOutcome === 'present'
      ? 'Present'
      : sessionOutcome === 'partial'
        ? 'Partial Attendance'
        : 'Absent';

  return (
    <div className="attendance-page">
      <h1 className="attendance-page__title">Attendance Verification</h1>

      <main className="attendance-page__main">
        <div className="attendance-page__content">
          <div className="attendance-page__webcam-section">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ display: webcamActive ? 'block' : 'none' }}
            />
            <canvas ref={canvasRef} className="attendance-page__canvas" />

            {!webcamActive && (
              <button className="attendance-page__open-camera-btn" onClick={startWebcam}>
                Open Camera
              </button>
            )}

            {webcamActive && !monitoringActive && !isVerifying && (
              <button
                className="attendance-page__verify-btn"
                onClick={() => void handleStartMonitoring()}
                disabled={!expectedStudentId}
              >
                Start Full-Session Monitoring
              </button>
            )}

            {webcamActive && monitoringActive && !isVerifying && (
              <button className="attendance-page__stop-btn" onClick={handleStopMonitoring}>
                Stop Monitoring
              </button>
            )}

            {activeChallenge && (
              <div className="attendance-page__challenge">
                <span className="attendance-page__challenge-badge">Live prompt</span>
                <p className="attendance-page__challenge-text">{activeChallenge}</p>
              </div>
            )}
          </div>

          <div className="attendance-page__result-section">
            <div className="attendance-page__session-card">
              <h2 className="attendance-page__session-title">Session Monitor</h2>
              <p className="attendance-page__session-message">{monitoringMessage}</p>

              <div className="attendance-page__session-metrics">
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Session length</span>
                  <strong>45 min</strong>
                </div>
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Time remaining</span>
                  <strong>{sessionEndsAt ? sessionTimeRemainingLabel : '--:--'}</strong>
                </div>
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Checks passed</span>
                  <strong>{successfulChecks}</strong>
                </div>
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Checks failed</span>
                  <strong>{failedChecks}</strong>
                </div>
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Checks run</span>
                  <strong>{checkNumber}</strong>
                </div>
                <div className="attendance-page__metric">
                  <span className="attendance-page__metric-label">Late-window check</span>
                  <strong>{lateWindowCheckPassed ? 'Passed' : 'Pending'}</strong>
                </div>
              </div>
            </div>

            {!expectedStudentId && (
              <p className="attendance-page__waiting">
                No student ID found for this account. Please sign in as a student first.
              </p>
            )}

            {!webcamActive && !result && !isVerifying && expectedStudentId && (
              <p className="attendance-page__waiting">
                Open the camera to begin full-session attendance monitoring.
              </p>
            )}

            {webcamActive && !monitoringActive && !isVerifying && !result && !sessionComplete && expectedStudentId && (
              <p className="attendance-page__waiting">
                Monitoring stays active for the entire session, with checks hidden at random times.
              </p>
            )}

            {isVerifying && (
              <div className="attendance-page__verifying">
                <div className="attendance-page__spinner" />
                <p>Running face verification...</p>
              </div>
            )}

            {sessionComplete && sessionOutcome && (
              <div
                className={`attendance-page__result-card ${
                  sessionOutcome === 'present'
                    ? 'attendance-page__result-card--present'
                    : 'attendance-page__result-card--absent'
                }`}
              >
                <span className="attendance-page__result-icon">
                  {sessionOutcome === 'present' ? 'PASS' : sessionOutcome === 'partial' ? 'PART' : 'FAIL'}
                </span>
                <h2 className="attendance-page__result-status">{outcomeLabel}</h2>
                <p className="attendance-page__result-detail">
                  Session verified for Student ID: {expectedStudentId}
                </p>
                <p className="attendance-page__result-similarity">
                  Passed {successfulChecks} of {checkNumber} checks.
                </p>
                <button
                  className="attendance-page__retry-btn"
                  onClick={() => void handleStartMonitoring()}
                >
                  Start New Session
                </button>
              </div>
            )}

            {result && !result.matched && !sessionComplete && (
              <div className="attendance-page__result-card attendance-page__result-card--absent">
                <span className="attendance-page__result-icon">FAIL</span>
                <h2 className="attendance-page__result-status">Check Failed</h2>
                <p className="attendance-page__result-detail">
                  {result.reason || 'Face did not match the logged-in student'}
                </p>
                {result.similarity !== undefined && (
                  <p className="attendance-page__result-similarity">
                    Best similarity: {(result.similarity * 100).toFixed(1)}%
                  </p>
                )}
                <button className="attendance-page__retry-btn" onClick={handleDismissResult}>
                  Dismiss
                </button>
              </div>
            )}

            {result && result.matched && !sessionComplete && (
              <div className="attendance-page__result-card attendance-page__result-card--present">
                <span className="attendance-page__result-icon">PASS</span>
                <h2 className="attendance-page__result-status">Check Passed</h2>
                <p className="attendance-page__result-detail">
                  Verified as Student ID: {result.student_id}
                </p>
                <p className="attendance-page__result-similarity">
                  Monitoring remains active until the session ends.
                </p>
                <button className="attendance-page__retry-btn" onClick={handleDismissResult}>
                  Hide Result
                </button>
              </div>
            )}

            {sessionStartedAt && !sessionComplete && (
              <p className="attendance-page__policy">
                Random re-checks continue for the full 45-minute session, including the last 8 minutes.
              </p>
            )}

            <p className="attendance-page__policy">
              Some live prompts are shown before checks. These prompts increase friction, but they are not yet computer-vision validated in this version.
            </p>
          </div>
        </div>
      </main>

      <button className="attendance-page__back-btn" onClick={() => navigate('/')}>
        Back to Dashboard
      </button>
    </div>
  );
};

export default AttendancePage;
