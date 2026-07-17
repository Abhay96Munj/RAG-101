import { useEffect, useState } from 'react'
import { askAgent } from '../api/client'
import { Answer } from '../components/Answer'
import { ErrorBanner } from '../components/ErrorBanner'
import { QuestionForm } from '../components/QuestionForm'
import { ToolCallCard } from '../components/ToolCallCard'
import { useTranscript } from '../hooks/useTranscript'

const EMPTY_STATE = (
  <p className="py-12 text-center text-sm text-gray-400">
    The agent picks its own tools — document search, calculator, datetime, knowledge-base listing — and can chain
    them. Try: “What did the resort cost, and what is 15% of that?”
  </p>
)

// Agent runs are multi-turn LLM loops and can take tens of seconds; a live
// counter keeps the wait honest where a bare spinner would look stuck.
function ElapsedTime() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(timer)
  }, [])
  return <span className="font-mono text-xs text-gray-400">{seconds}s</span>
}

export function AgentView() {
  const { exchanges, pending, ask } = useTranscript(askAgent)

  return (
    <div className="space-y-4">
      <QuestionForm placeholder="e.g. What did the resort cost, and what is 15% of that?" pending={pending} onAsk={ask} />

      {exchanges.length === 0 ? EMPTY_STATE : null}

      <ol className="space-y-4">
        {exchanges.map((ex) => (
          <li key={ex.id} className="space-y-2">
            <p className="ml-auto w-fit max-w-[85%] rounded-lg bg-indigo-600 px-3 py-2 text-sm text-white">
              {ex.question}
            </p>
            <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
              {ex.status === 'pending' ? (
                <p className="animate-pulse text-sm text-gray-400">
                  Agent is thinking and calling tools… <ElapsedTime />
                </p>
              ) : ex.status === 'error' ? (
                <ErrorBanner message={ex.error ?? 'Unknown error'} />
              ) : (
                <div className="space-y-2">
                  {ex.response!.tool_calls.length > 0 ? (
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
                        Tool calls ({ex.response!.tool_calls.length})
                      </p>
                      {ex.response!.tool_calls.map((call, i) => (
                        <ToolCallCard key={i} step={i + 1} call={call} />
                      ))}
                    </div>
                  ) : null}
                  <Answer text={ex.response!.answer} refused={false} />
                  <p className="font-mono text-[10px] text-gray-300">query_id: {ex.response!.query_id}</p>
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
