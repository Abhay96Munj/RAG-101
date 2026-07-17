import { useState } from 'react'

interface Props {
  placeholder: string
  pending: boolean
  onAsk: (question: string) => void
}

export function QuestionForm({ placeholder, pending, onAsk }: Props) {
  const [question, setQuestion] = useState('')

  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        const trimmed = question.trim()
        if (trimmed === '' || pending) return
        onAsk(trimmed)
        setQuestion('')
      }}
    >
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={placeholder}
        className="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
      />
      <button
        type="submit"
        disabled={pending || question.trim() === ''}
        className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {pending ? 'Asking…' : 'Ask'}
      </button>
    </form>
  )
}
