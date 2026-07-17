import { useRef, useState } from 'react'
import useSWR from 'swr'
import { fetchDocuments, uploadPdf } from '../api/client'
import { ErrorBanner } from '../components/ErrorBanner'

interface UploadLogEntry {
  id: number
  filename: string
  status: 'indexing' | 'done' | 'error'
  detail?: string
}

export function DocumentsView() {
  // SWR dedupes and caches the listing; mutate() after a successful upload
  // refetches so the table reflects the new document immediately.
  const { data, error, isLoading, mutate } = useSWR('documents', fetchDocuments)
  const [uploads, setUploads] = useState<UploadLogEntry[]>([])
  const nextId = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)

  const uploading = uploads.some((u) => u.status === 'indexing')

  function onFileChosen(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploads((prev) => [...prev, { id: nextId.current++, filename: file.name, status: 'error', detail: 'Only .pdf files are supported.' }])
      return
    }
    const id = nextId.current++
    setUploads((prev) => [...prev, { id, filename: file.name, status: 'indexing' }])
    uploadPdf(file)
      .then((res) => {
        setUploads((prev) =>
          prev.map((u) => (u.id === id ? { ...u, status: 'done', detail: `${res.chunks_added} chunks indexed` } : u)),
        )
        void mutate()
      })
      .catch((err: unknown) =>
        setUploads((prev) =>
          prev.map((u) =>
            u.id === id ? { ...u, status: 'error', detail: err instanceof Error ? err.message : String(err) } : u,
          ),
        ),
      )
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border-2 border-dashed border-gray-300 bg-white p-6 text-center">
        <p className="mb-3 text-sm text-gray-500">
          Upload a PDF — it gets chunked, embedded, and indexed before the response returns, so large files take a
          while.
        </p>
        <input
          ref={fileInput}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file !== undefined) onFileChosen(file)
            e.target.value = '' // allow re-selecting after an error
          }}
        />
        <button
          type="button"
          disabled={uploading}
          onClick={() => fileInput.current?.click()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {uploading ? 'Indexing…' : 'Choose PDF'}
        </button>
      </div>

      {uploads.length > 0 ? (
        <ul className="space-y-1.5">
          {uploads.map((u) => (
            <li key={u.id} className="flex items-center gap-2 text-sm">
              <span
                className={
                  u.status === 'done' ? 'text-green-600' : u.status === 'error' ? 'text-red-600' : 'animate-pulse text-gray-400'
                }
              >
                {u.status === 'done' ? '✓' : u.status === 'error' ? '✕' : '…'}
              </span>
              <span className="font-medium text-gray-700">{u.filename}</span>
              <span className="text-gray-500">
                {u.status === 'indexing' ? 'uploading → indexing…' : (u.detail ?? '')}
              </span>
            </li>
          ))}
        </ul>
      ) : null}

      <div>
        <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400">Knowledge base</h2>
        {isLoading ? (
          <p className="animate-pulse text-sm text-gray-400">Loading…</p>
        ) : error !== undefined ? (
          <ErrorBanner message={error instanceof Error ? error.message : 'Failed to load documents'} />
        ) : data === undefined || data.documents.length === 0 ? (
          <p className="text-sm text-gray-400">The knowledge base is empty — upload a PDF to get started.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-gray-200 text-xs uppercase tracking-wide text-gray-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Document</th>
                  <th className="px-3 py-2 font-medium">Chunks</th>
                  <th className="px-3 py-2 font-medium">Pages</th>
                </tr>
              </thead>
              <tbody>
                {data.documents.map((doc) => (
                  <tr key={doc.filename} className="border-b border-gray-100 last:border-0">
                    <td className="px-3 py-2 font-medium text-gray-700">{doc.filename}</td>
                    <td className="px-3 py-2 text-gray-500">{doc.chunk_count}</td>
                    <td className="px-3 py-2 text-gray-500">{doc.pages > 0 ? doc.pages : '—'}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="text-xs text-gray-400">
                  <td className="px-3 py-2">{data.documents.length} document(s)</td>
                  <td className="px-3 py-2">{data.total_chunks} total</td>
                  <td />
                </tr>
              </tfoot>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
