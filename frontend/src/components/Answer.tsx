import { Suspense, lazy } from 'react'

// react-markdown is the heaviest dependency — keep it out of the initial
// bundle and only load it when the first answer renders.
const Markdown = lazy(() => import('react-markdown'))

interface Props {
  text: string
  refused: boolean
}

export function Answer({ text, refused }: Props) {
  return refused ? (
    <p className="text-sm italic text-gray-500">{text}</p>
  ) : (
    <div className="text-sm leading-relaxed text-gray-800 [&_a]:text-indigo-600 [&_a]:underline [&_code]:rounded [&_code]:bg-gray-100 [&_code]:px-1 [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-1.5 [&_ul]:list-disc [&_ul]:pl-5">
      <Suspense fallback={<p className="whitespace-pre-wrap">{text}</p>}>
        <Markdown>{text}</Markdown>
      </Suspense>
    </div>
  )
}
