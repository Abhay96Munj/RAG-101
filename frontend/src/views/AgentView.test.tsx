import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { AgentResponse } from '../api/types'
import { AgentView } from './AgentView'

vi.mock('../api/client', () => ({
  askAgent: vi.fn(),
}))

import { askAgent } from '../api/client'

describe('AgentView', () => {
  it('renders the answer and the ordered tool-call trace', async () => {
    const response: AgentResponse = {
      answer: '15% of $700M is $105M.',
      tool_calls: [
        { tool_name: 'search_documents', arguments: { query: 'resort cost' }, result: 'The resort cost $700M...' },
        { tool_name: 'calculator', arguments: { expression: '700 * 0.15' }, result: '105.0' },
      ],
      query_id: 'a1b2c3d4',
    }
    vi.mocked(askAgent).mockResolvedValue(response)

    render(<AgentView />)
    fireEvent.change(screen.getByPlaceholderText(/15%/), { target: { value: 'What is 15% of the resort cost?' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))

    expect(await screen.findByText('15% of $700M is $105M.')).toBeInTheDocument()
    expect(screen.getByText('search_documents')).toBeInTheDocument()
    expect(screen.getByText('calculator')).toBeInTheDocument()
    expect(screen.getByText(/Tool calls \(2\)/)).toBeInTheDocument()
  })
})
