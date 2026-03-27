import type { CapturedFrame } from '../types/frame';
import { apiPost } from './api.ts';

export const registerFaces = (
  frames: CapturedFrame[],
  studentId: string
): Promise<void> => {
  return apiPost('/student/register-faces', { frames, studentId });
};

export interface VerifyFaceResult {
  matched: boolean;
  student_id?: string;
  similarity?: number;
  reason?: string;
}

export const verifyFace = (imageData: string): Promise<VerifyFaceResult> => {
  return apiPost('/student/verify-face', { imageData });
};