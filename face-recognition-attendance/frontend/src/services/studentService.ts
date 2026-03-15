import type { CapturedFrame } from '../types/frame';
import { apiPost } from './api.ts';

export const registerFaces = (
  frames: CapturedFrame[],
  studentId: string
): Promise<void> => {
  return apiPost('/student/register-faces', { frames, studentId });
};