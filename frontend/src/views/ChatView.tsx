import { askChat } from '../api/client'
import { Answer } from '../components/Answer'
import { ErrorBanner } from '../components/ErrorBanner'
import { QuestionForm } from '../components/QuestionForm'
import { SourceCard } from '../components/SourceCard'
import { useTranscript } from '../hooks/useTranscript'

const EMPTY_STATE = (
  <p className="py-12 text-center text-sm text-gray-400">
    Ask a question about your uploaded documents — the answer comes back with its source chunks.
  </p>
)

export function ChatView() {
  const { exchanges, pending, ask } = useTranscript(askChat)

  return (
    <div className="space-y-4">
      <QuestionForm placeholder="e.g. How much did the Sunseeker Resort cost to build?" pending={pending} onAsk={ask} />

      {exchanges.length === 0 ? EMPTY_STATE : null}

      <ol className="space-y-4">
        {exchanges.map((ex) => (
          <li key={ex.id} className="space-y-2">
            <p className="ml-auto w-fit max-w-[85%] rounded-lg bg-indigo-600 px-3 py-2 text-sm text-white">
              {ex.question}
            </p>
            <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
              {ex.status === 'pending' ? (
                <p className="animate-pulse text-sm text-gray-400">Retrieving and generating…</p>
              ) : ex.status === 'error' ? (
                <ErrorBanner message={ex.error ?? 'Unknown error'} />
              ) : (
                <div className="space-y-2">
                  <Answer text={ex.response!.answer} refused={ex.response!.refused} />
                  {ex.response!.sources.length > 0 ? (
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
                        Sources ({ex.response!.sources.length})
                      </p>
                      {ex.response!.sources.map((chunk) => (
                        <SourceCard key={`${chunk.source}-${chunk.chunk_index}`} chunk={chunk} />
                      ))}
                    </div>
                  ) : null}
                  {ex.response!.query_id !== null ? (
                    <p className="font-mono text-[10px] text-gray-300">query_id: {ex.response!.query_id}</p>
                  ) : null}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
