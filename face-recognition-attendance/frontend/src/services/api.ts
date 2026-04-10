const BASE_URL = 'http://localhost:8000/api';

const getAuthHeaders = (): HeadersInit => {
  const rawUser = localStorage.getItem('user');
  if (!rawUser) return {};

  try {
    const user = JSON.parse(rawUser);
    return user.email ? { 'x-user-email': user.email } : {};
  } catch {
    return {};
  }
};

export const apiPost = async <T>(endpoint: string, body: unknown): Promise<T> => {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
};

export const apiGet = async <T>(endpoint: string): Promise<T> => {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
};
