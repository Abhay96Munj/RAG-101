import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ChatResponse } from '../api/types'
import { ChatView } from './ChatView'

vi.mock('../api/client', () => ({
  askChat: vi.fn(),
}))

import { askChat } from '../api/client'

function ask(question: string) {
  fireEvent.change(screen.getByPlaceholderText(/Sunseeker/), { target: { value: question } })
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }))
}

describe('ChatView', () => {
  it('renders an answer with its source chunks', async () => {
    const response: ChatResponse = {
      answer: 'It cost $700 million.',
      sources: [
        { source: 'resort.pdf', page_num: 3, chunk_index: 7, score: 0.81, rerank_score: 4.2, text: 'The resort cost...' },
      ],
      refused: false,
      query_id: 'abc12345',
    }
    vi.mocked(askChat).mockResolvedValue(response)

    render(<ChatView />)
    ask('How much did the resort cost?')

    expect(await screen.findByText('It cost $700 million.')).toBeInTheDocument()
    expect(screen.getByText('resort.pdf')).toBeInTheDocument()
    expect(screen.getByText(/query_id: abc12345/)).toBeInTheDocument()
  })

  it('renders a refusal distinctly, without a sources section', async () => {
    const response: ChatResponse = {
      answer: 'I don’t have enough information — nothing relevant was found.',
      sources: [],
      refused: true,
      query_id: 'def67890',
    }
    vi.mocked(askChat).mockResolvedValue(response)

    render(<ChatView />)
    ask('Who won the 1962 world cup?')

    expect(await screen.findByText(/nothing relevant was found/)).toBeInTheDocument()
    expect(screen.queryByText(/Sources \(/)).not.toBeInTheDocument()
  })
})
