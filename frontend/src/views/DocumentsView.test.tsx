import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { DocumentsResponse } from '../api/types'
import { DocumentsView } from './DocumentsView'

vi.mock('../api/client', () => ({
  fetchDocuments: vi.fn(),
  uploadPdf: vi.fn(),
}))

import { fetchDocuments } from '../api/client'

describe('DocumentsView', () => {
  it('renders the knowledge-base table from GET /documents', async () => {
    const response: DocumentsResponse = {
      documents: [
        { filename: 'resort.pdf', chunk_count: 42, pages: 12 },
        { filename: 'blackstone.pdf', chunk_count: 7, pages: 3 },
      ],
      total_chunks: 49,
    }
    vi.mocked(fetchDocuments).mockResolvedValue(response)

    render(<DocumentsView />)

    expect(await screen.findByText('resort.pdf')).toBeInTheDocument()
    expect(screen.getByText('blackstone.pdf')).toBeInTheDocument()
    expect(screen.getByText('49 total')).toBeInTheDocument()
  })
})
