import type { SourceChunk } from '../api/types'

interface Props {
  chunk: SourceChunk
}

export function SourceCard({ chunk }: Props) {
  return (
    <details className="rounded-lg border border-gray-200 bg-gray-50 text-sm">
      <summary className="flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 text-gray-700">
        <span className="font-medium">{chunk.source}</span>
        {chunk.page_num !== null ? <span className="text-gray-500">p. {chunk.page_num}</span> : null}
        <span className="ml-auto font-mono text-xs text-gray-400">
          rerank {chunk.rerank_score.toFixed(3)} · retr {chunk.score.toFixed(3)} · #{chunk.chunk_index}
        </span>
      </summary>
      <p className="whitespace-pre-wrap border-t border-gray-200 px-3 py-2 text-xs leading-relaxed text-gray-600">
        {chunk.text}
      </p>
    </details>
  )
}
