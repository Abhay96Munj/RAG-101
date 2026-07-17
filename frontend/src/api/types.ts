// TypeScript mirrors of the Pydantic response models in app/schemas/.
// Keep in sync by hand — the shapes are small and change rarely.

export interface SourceChunk {
  source: string
  page_num: number | null // older chunks in the store may not have it
  chunk_index: number
  score: number
  rerank_score: number
  text: string
}

export interface ChatResponse {
  answer: string
  sources: SourceChunk[]
  refused: boolean // true → nothing relevant found, no LLM call was made
  query_id: string | null
}

export interface ToolCallLog {
  tool_name: string
  arguments: Record<string, unknown>
  result: string
}

export interface AgentResponse {
  answer: string
  tool_calls: ToolCallLog[]
  query_id: string
}

export interface DocumentInfo {
  filename: string
  chunk_count: number
  pages: number // distinct page numbers; 0 if older chunks lack page_num
}

export interface DocumentsResponse {
  documents: DocumentInfo[]
  total_chunks: number
}

export interface UploadResponse {
  message: string
  filename: string
  chunks_added: number
}
