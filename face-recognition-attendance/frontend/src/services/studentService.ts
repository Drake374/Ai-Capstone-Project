import type { CapturedFrame } from '../types/frame';
import { apiPost, apiGet } from './api.ts';

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

export const verifyFace = (
  imageData: string,
  expectedStudentId?: string
): Promise<VerifyFaceResult> => {
  return apiPost('/student/verify-face', { imageData, expectedStudentId });
};

export interface RegisterStudentResult {
  student_id: string;
  name: string;
  email: string;
  role: 'student' | 'admin';
  registered: boolean;
}

export interface RegisterAdminResult {
  name: string;
  email: string;
  photo_url: string;
  role: 'admin';
}

export const registerStudent = (
  studentId: string,
  name: string,
  email: string,
  photoUrl: string = ''
): Promise<RegisterStudentResult> => {
  return apiPost('/student/register-student', { studentId, name, email, photoUrl });
};

export const registerAdmin = (
  name: string,
  email: string,
  photoUrl: string = ''
): Promise<RegisterAdminResult> => {
  return apiPost('/student/register-admin', { name, email, photoUrl });
};

export interface StudentProfile {
  found: boolean;
  role?: 'student' | 'admin';
  student_id?: string;
  name?: string;
  email?: string;
  photo_url?: string;
  registered?: boolean;
  face_count?: number;
}

export const getStudentProfile = (email: string): Promise<StudentProfile> => {
  return apiGet(`/student/profile?email=${encodeURIComponent(email)}`);
};
