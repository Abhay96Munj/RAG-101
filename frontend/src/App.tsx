import { useState } from 'react'
import { AgentView } from './views/AgentView'
import { ChatView } from './views/ChatView'
import { DocumentsView } from './views/DocumentsView'

type Tab = 'chat' | 'agent' | 'documents'

const TABS: { key: Tab; label: string }[] = [
  { key: 'chat', label: 'Chat' },
  { key: 'agent', label: 'Agent' },
  { key: 'documents', label: 'Documents' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-3xl items-center gap-6 px-4 py-3">
          <h1 className="text-base font-semibold text-gray-800">RAG-101</h1>
          <nav className="flex gap-1">
            {TABS.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => setTab(key)}
                className={
                  tab === key
                    ? 'rounded-lg bg-indigo-50 px-3 py-1.5 text-sm font-medium text-indigo-700'
                    : 'rounded-lg px-3 py-1.5 text-sm text-gray-500 hover:bg-gray-100'
                }
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* All three views stay mounted so transcripts and SWR state survive
          tab switches — only visibility toggles. */}
      <main className="mx-auto max-w-3xl px-4 py-6">
        <div className={tab === 'chat' ? '' : 'hidden'}>
          <ChatView />
        </div>
        <div className={tab === 'agent' ? '' : 'hidden'}>
          <AgentView />
        </div>
        <div className={tab === 'documents' ? '' : 'hidden'}>
          <DocumentsView />
        </div>
      </main>
    </div>
  )
}
