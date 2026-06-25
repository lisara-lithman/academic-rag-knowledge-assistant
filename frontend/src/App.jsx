// frontend/src/App.jsx
import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { sendMessage } from './api.js'
import './App.css'

// ─────────────────────────────────────────────────────────────
// COMPONENT: A single chat message bubble
// ─────────────────────────────────────────────────────────────
function Message({ role, content, chunks, isStreaming }) {
  const isUser = role === 'user'

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      {/* Avatar */}
      <div className={`avatar ${isUser ? 'avatar-user' : 'avatar-bot'}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      <div className="message-content">
        {/* The bubble */}
        <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
          {isUser ? (
            <p>{content}</p>
          ) : (
            <div className="markdown">
              <ReactMarkdown>{content}</ReactMarkdown>
              {isStreaming && <span className="cursor">▋</span>}
            </div>
          )}
        </div>

        {/* Source chunks — only shown after streaming is done */}
        {!isUser && !isStreaming && chunks && chunks.length > 0 && (
          <Sources chunks={chunks} />
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// COMPONENT: Collapsible source references
// ─────────────────────────────────────────────────────────────
function Sources({ chunks }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        📚 {chunks.length} source{chunks.length > 1 ? 's' : ''} used
        <span className={`chevron ${open ? 'open' : ''}`}>▾</span>
      </button>
      {open && (
        <div className="sources-list">
          {chunks.map((chunk, i) => (
            <div key={i} className="source-item">
              <div className="source-meta">
                <span className="source-file">
                  📄 {chunk.metadata?.source || 'Unknown'}
                </span>
                {chunk.metadata?.page && (
                  <span className="source-page">Page {chunk.metadata.page}</span>
                )}
                {chunk.score !== undefined && (
                  <span className="source-score">
                    {(chunk.score * 100).toFixed(0)}% match
                  </span>
                )}
              </div>
              <p className="source-text">{chunk.text?.slice(0, 200)}…</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// COMPONENT: Typing indicator (3 bouncing dots)
// ─────────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="message-row bot-row">
      <div className="avatar avatar-bot">🤖</div>
      <div className="bubble bubble-bot typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      content: "Hello! I'm your **AI Tutor** 🎓\n\nAsk me anything about **Operating Systems and System Administration** — I'll find the most relevant material from your lecture notes and explain it clearly.",
      chunks: [],
    }
  ])

  const [input, setInput]         = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef(null)

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Handle Send ──────────────────────────────────────────────
  async function handleSend() {
    const text = input.trim()
    if (!text || isLoading) return

    // 1. Add the user message to chat
    const userMsg = { role: 'user', content: text, chunks: [] }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    // 2. Build history for the backend: [ ["user", "bot"], ... ]
    // Use `messages` (before adding userMsg) so the current question is NOT in history.
    // Only include complete pairs where the bot reply is done streaming.
    const history = []
    for (let i = 0; i < messages.length - 1; i++) {
      const curr = messages[i]
      const next = messages[i + 1]
      if (curr.role === 'user' && next?.role === 'bot' && !next.isStreaming && next.content) {
        history.push([curr.content, next.content])
        i++ // skip the bot message we just paired
      }
    }

    // 3. Add an empty bot message placeholder
    setMessages(prev => [
      ...prev,
      { role: 'bot', content: '', chunks: [], isStreaming: true }
    ])

    // 4. Stream the response
    try {
      let metadata = null

      await sendMessage(
        text,
        history,

        // onToken: append each word to the last bot message
        (token) => {
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.content += token
            updated[updated.length - 1] = last
            return updated
          })
          setIsLoading(false)
        },

        // onMetadata: store decision + source chunks
        (meta) => {
          metadata = meta
        },

        // onDone: finalize the message
        () => {
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.isStreaming = false
            last.chunks = metadata?.chunks || []
            updated[updated.length - 1] = last
            return updated
          })
          setIsLoading(false)
        }
      )
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'bot',
          content: '❌ **Error:** Could not reach the server. Is your backend running?\n\n```\npython src/server.py\n```',
          chunks: [],
          isStreaming: false
        }
        return updated
      })
      setIsLoading(false)
    }
  }

  // Send on Enter, new line on Shift+Enter
  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // ── Render ────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <div className="logo">🎓</div>
          <div>
            <h1 className="header-title">AI Tutor</h1>
            <p className="header-subtitle">Academic RAG Knowledge Assistant</p>
          </div>
        </div>
        <div className="header-status">
          <span className="status-dot"></span>
          <span>Connected</span>
        </div>
      </header>

      {/* ── Chat Messages ── */}
      <main className="chat-area">
        {messages.map((msg, i) => (
          <Message
            key={i}
            role={msg.role}
            content={msg.content}
            chunks={msg.chunks}
            isStreaming={msg.isStreaming}
          />
        ))}

        {/* Typing indicator — shown while waiting for first token */}
        {isLoading && messages[messages.length - 1]?.role !== 'bot' && (
          <TypingIndicator />
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </main>

      {/* ── Input Bar ── */}
      <footer className="input-area">
        <div className="input-wrapper">
          <textarea
            className="input-box"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about processes, scheduling, file systems, memory management..."
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </div>
        <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
      </footer>

    </div>
  )
}
