import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, askChat } from './client'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('api client error normalization', () => {
  it('turns a FastAPI {detail} error into an ApiError with that message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'LLM generation failed: boom' }), {
          status: 502,
          statusText: 'Bad Gateway',
        }),
      ),
    )
    const err = await askChat('q').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(502)
    expect((err as ApiError).message).toBe('LLM generation failed: boom')
  })

  it('falls back to the status line when the error body is not JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('<html>oops</html>', { status: 500, statusText: 'Internal Server Error' })),
    )
    const err = await askChat('q').catch((e: unknown) => e)
    expect((err as ApiError).message).toBe('500 Internal Server Error')
  })

  it('maps a network failure to ApiError with status 0', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const err = await askChat('q').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(0)
  })
})
