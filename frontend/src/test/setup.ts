import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// RTL's automatic cleanup needs a global afterEach; we run Vitest with
// globals disabled, so register it explicitly.
afterEach(cleanup)
