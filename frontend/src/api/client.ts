import type { AgentResponse, ChatResponse, DocumentsResponse, UploadResponse } from './types'

const BASE = '/api/v1'

// Every failure — HTTP error or network-down — reaches the views as an
// ApiError, so components never have to unpick raw fetch failures.
// status 0 means the server couldn't be reached at all.
export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'Cannot reach the server — is the API running on port 8000?')
  }
  if (!res.ok) {
    // FastAPI errors carry {detail: string}; fall back to the status line.
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      /* non-JSON body — keep the status line */
    }
    throw new ApiError(res.status, message)
  }
  return res.json()
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function askChat(question: string): Promise<ChatResponse> {
  return postJson('/chat/query', { question })
}

export function askAgent(question: string): Promise<AgentResponse> {
  return postJson('/agent/query', { question })
}

export function fetchDocuments(): Promise<DocumentsResponse> {
  return request('/documents')
}

export function uploadPdf(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  // No Content-Type header — the browser sets the multipart boundary itself.
  return request('/ingest/upload', { method: 'POST', body: form })
}
