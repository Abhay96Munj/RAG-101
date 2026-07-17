import { useRef, useState } from 'react'

export interface Exchange<R> {
  id: number
  question: string
  status: 'pending' | 'done' | 'error'
  response?: R
  error?: string
}

// Shared transcript state for the Chat and Agent views: an append-only list
// of question → result exchanges. Each ask() is independent (the backend is
// stateless per question) — this is a session log, not conversation memory.
export function useTranscript<R>(send: (question: string) => Promise<R>) {
  const [exchanges, setExchanges] = useState<Exchange<R>[]>([])
  const nextId = useRef(0)

  const pending = exchanges.some((e) => e.status === 'pending')

  function ask(question: string) {
    const id = nextId.current++
    setExchanges((prev) => [...prev, { id, question, status: 'pending' }])
    send(question)
      .then((response) =>
        setExchanges((prev) => prev.map((e) => (e.id === id ? { ...e, status: 'done', response } : e))),
      )
      .catch((err: unknown) =>
        setExchanges((prev) =>
          prev.map((e) =>
            e.id === id ? { ...e, status: 'error', error: err instanceof Error ? err.message : String(err) } : e,
          ),
        ),
      )
  }

  return { exchanges, pending, ask }
}
