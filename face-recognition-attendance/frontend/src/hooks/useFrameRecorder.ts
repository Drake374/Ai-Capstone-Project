import { useRef, useState, useCallback } from 'react';
import type { CapturedFrame } from '../types/frame';
import { registerFaces } from '../services/studentService';

export const useFrameRecorder = (
  videoRef: React.RefObject<HTMLVideoElement | null>
) => {
  const [isRecording, setIsRecording] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<CapturedFrame[]>([]);
  const [previewMode, setPreviewMode] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recordingIntervalRef = useRef<number | null>(null);

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
      timestamp: Date.now(),
    };
  }, [videoRef]);

  const startRecording = useCallback(() => {
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
  }, [captureFrame]);

  const cancelRecording = useCallback(() => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    setIsRecording(false);
    setCapturedFrames([]);
    setPreviewMode(false);
  }, []);

  const confirmFrames = useCallback(async () => {
    try {
      await registerFaces(capturedFrames, 'current-student-id');
      setCapturedFrames([]);
      setPreviewMode(false);
    } catch (error) {
      console.error('Save failed:', error);
    }
  }, [capturedFrames]);

  return {
    isRecording,
    capturedFrames,
    previewMode,
    canvasRef,
    startRecording,
    cancelRecording,
    confirmFrames,
  };
};