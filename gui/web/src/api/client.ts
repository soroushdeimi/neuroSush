import { ApiError, ApiErrorBody } from './types';

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    if (res.status === 204) return undefined as T;
    return res.json();
  }
  let errBody: unknown;
  try {
    errBody = await res.json();
  } catch {
    // Non-JSON error response
  }
  const errorInfo = (errBody as ApiErrorBody)?.error || {
    code: 'unknown_error',
    message: res.statusText || 'Request failed',
    issues: [],
  };
  throw new ApiError(res.status, errorInfo.code, errorInfo.message, errorInfo.issues);
}

export async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  return handleResponse<T>(res);
}

export async function post<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return handleResponse<T>(res);
}

export async function put<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return handleResponse<T>(res);
}

export async function del<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'DELETE' });
  return handleResponse<T>(res);
}
