/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev proxy is the whole CORS story: the browser only ever talks to the
// Vite origin, which forwards /api/* to the FastAPI server. No backend
// CORS configuration needed (see docs/specs/2026-07-17-frontend-design.md §6.1).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
