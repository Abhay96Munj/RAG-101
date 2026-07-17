import type { ToolCallLog } from '../api/types'

interface Props {
  step: number
  call: ToolCallLog
}

export function ToolCallCard({ step, call }: Props) {
  return (
    <details className="rounded-lg border border-gray-200 bg-gray-50 text-sm">
      <summary className="flex cursor-pointer items-center gap-2 px-3 py-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-xs font-semibold text-indigo-700">
          {step}
        </span>
        <span className="font-mono text-xs font-medium text-gray-700">{call.tool_name}</span>
        <span className="truncate font-mono text-xs text-gray-400">{JSON.stringify(call.arguments)}</span>
      </summary>
      <pre className="overflow-x-auto whitespace-pre-wrap border-t border-gray-200 px-3 py-2 font-mono text-xs leading-relaxed text-gray-600">
        {call.result}
      </pre>
    </details>
  )
}
