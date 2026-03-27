import { apiGet } from './api';

const BASE_URL = 'http://localhost:8000/api';

export interface AttendanceLogEntry {
  student_id: string;
  status: string;
  similarity: number;
  timestamp: string;
}

export const getAttendanceLogs = (
  startDate?: string,
  endDate?: string
): Promise<AttendanceLogEntry[]> => {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);

  const query = params.toString() ? `?${params.toString()}` : '';
  return apiGet(`/admin/attendance-logs${query}`);
};

export const exportAttendanceCsv = async (
  startDate?: string,
  endDate?: string
): Promise<void> => {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);

  const query = params.toString() ? `?${params.toString()}` : '';
  const url = `${BASE_URL}/admin/attendance-logs/export${query}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Export failed: ${response.status}`);
  }

  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;

  // Extract filename from Content-Disposition header or use default
  const disposition = response.headers.get('Content-Disposition');
  const match = disposition?.match(/filename=(.+)/);
  a.download = match ? match[1] : 'attendance_logs.csv';

  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(downloadUrl);
};
