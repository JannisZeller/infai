import { useEffect, useMemo, useRef, useState } from 'react'

const DECISION_DEFAULTS = {}

function createAssistantTurn() {
  return {
    id: crypto.randomUUID(),
    kind: 'assistant',
    thinking: '',
    response: '',
    toolCalls: [],
    appLaunches: [],
  }
}

async function ensureSession() {
  await fetch('/api/session', { credentials: 'include' })
}

async function readNdjsonStream(response, onItem) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.trim()) continue
      onItem(JSON.parse(line))
    }
  }

  if (buffer.trim()) {
    onItem(JSON.parse(buffer))
  }
}

function isSafeHttpUrl(value) {
  return typeof value === 'string' && (value.startsWith('http://') || value.startsWith('https://'))
}

function ensureAssistantTurn(transcript) {
  const nextTranscript = [...transcript]
  const lastItem = nextTranscript.at(-1)
  if (lastItem?.kind !== 'assistant') {
    nextTranscript.push(createAssistantTurn())
  }
  return nextTranscript
}

function updateAssistantTurn(transcript, update) {
  const nextTranscript = ensureAssistantTurn(transcript)
  const assistant = nextTranscript.at(-1)
  nextTranscript[nextTranscript.length - 1] = update(assistant)
  return nextTranscript
}

function upsertToolCall(toolCalls, nextToolCall) {
  const existingIndex = toolCalls.findIndex((toolCall) => toolCall.toolCallId === nextToolCall.toolCallId)
  if (existingIndex === -1) {
    return [...toolCalls, nextToolCall]
  }

  const merged = [...toolCalls]
  merged[existingIndex] = { ...merged[existingIndex], ...nextToolCall }
  return merged
}

function updateTranscriptForStreamItem(transcript, item) {
  switch (item.type) {
    case 'user_prompt':
    case 'system_prompt':
    case 'part_start':
    case 'stream_end':
      return transcript
    case 'thinking_delta':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        thinking: `${assistant.thinking}${item.delta}`,
      }))
    case 'model_response_delta':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        response: `${assistant.response}${item.delta}`,
      }))
    case 'thinking_step':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        thinking: assistant.thinking || item.thoughts,
      }))
    case 'model_response':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        response: assistant.response || item.response,
      }))
    case 'tool_call':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        toolCalls: upsertToolCall(assistant.toolCalls, {
          toolCallId: item.tool_call_id,
          toolName: item.tool_name,
          args: item.args,
          status: 'running',
          result: null,
          isRetry: false,
        }),
      }))
    case 'tool_result':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        toolCalls: upsertToolCall(assistant.toolCalls, {
          toolCallId: item.tool_call_id,
          toolName: item.tool_name,
          args: null,
          status: item.is_retry ? 'retry' : 'completed',
          result: item.result,
          isRetry: item.is_retry,
        }),
      }))
    case 'mcp_app_launch_request':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        appLaunches: assistant.appLaunches.some((launch) => launch.toolCallId === item.tool_call_id)
          ? assistant.appLaunches
          : [
              ...assistant.appLaunches,
              {
                toolCallId: item.tool_call_id,
                toolName: item.tool_name,
                title: item.title,
                url: item.url,
              },
            ],
      }))
    case 'tool_approval_request':
      return updateAssistantTurn(transcript, (assistant) => ({
        ...assistant,
        toolCalls: upsertToolCall(assistant.toolCalls, {
          toolCallId: item.tool_call_id,
          toolName: item.tool_name,
          args: item.args,
          status: 'awaiting_approval',
          result: null,
          isRetry: false,
        }),
      }))
    default:
      return transcript
  }
}

function formatToolStatus(status) {
  if (status === 'awaiting_approval') return 'Awaiting approval'
  if (status === 'completed') return 'Completed'
  if (status === 'retry') return 'Retry requested'
  return 'Running'
}

function renderValue(value) {
  if (value == null) return 'null'
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [transcript, setTranscript] = useState([])
  const [pendingApprovals, setPendingApprovals] = useState([])
  const [approvalDecisions, setApprovalDecisions] = useState(DECISION_DEFAULTS)
  const [currentElicitation, setCurrentElicitation] = useState(null)
  const [elicitationValues, setElicitationValues] = useState({})
  const [currentAuth, setCurrentAuth] = useState(null)
  const [activeApp, setActiveApp] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const eventSourceRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    ensureSession()
      .then(() => {
        if (cancelled) return
        const eventSource = new EventSource('/api/events', { withCredentials: true })
        eventSource.onmessage = (message) => {
          const event = JSON.parse(message.data)
          if (event.type === 'mcp_elicitation_request') {
            setCurrentElicitation(event)
            setElicitationValues({})
          }
          if (event.type === 'mcp_auth_required') {
            setCurrentAuth(event)
            if (isSafeHttpUrl(event.authorization_url)) {
              window.open(event.authorization_url, '_blank', 'popup,width=600,height=800')
            }
          }
          if (event.type === 'mcp_auth_completed' || event.type === 'mcp_auth_failed') {
            setCurrentAuth(null)
            if (event.message) {
              setStatusMessage(event.message)
            }
          }
        }
        eventSourceRef.current = eventSource
      })
      .catch(() => {
        setStatusMessage('Unable to initialize the browser session.')
      })

    return () => {
      cancelled = true
      eventSourceRef.current?.close()
    }
  }, [])

  const canResume = useMemo(() => pendingApprovals.length > 0, [pendingApprovals])

  async function handlePromptSubmit(event) {
    event.preventDefault()
    if (!prompt.trim() || isStreaming) return

    try {
      setStatusMessage('')
      setIsStreaming(true)
      setPendingApprovals([])
      setApprovalDecisions(DECISION_DEFAULTS)
      setTranscript((items) => [...items, { id: crypto.randomUUID(), kind: 'user', text: prompt }, createAssistantTurn()])

      const response = await fetch('/api/chat/turn', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })
      if (!response.ok) {
        throw new Error(`Chat request failed (${response.status})`)
      }

      setPrompt('')
      await readNdjsonStream(response, handleStreamItem)
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Chat request failed.')
    } finally {
      setIsStreaming(false)
    }
  }

  function handleStreamItem(item) {
    if (item.type === 'tool_approval_request') {
      setPendingApprovals((items) => [...items, item])
      setApprovalDecisions((current) => ({ ...current, [item.tool_call_id]: true }))
    }

    if (item.type === 'mcp_app_launch_request') {
      setActiveApp(item)
    }

    setTranscript((items) => updateTranscriptForStreamItem(items, item))
  }

  async function handleResume() {
    if (!pendingApprovals.length || isStreaming) return

    try {
      setStatusMessage('')
      setIsStreaming(true)
      const resumeToken = pendingApprovals[0].resume_token
      const approvals = pendingApprovals.map((request) => ({
        tool_call_id: request.tool_call_id,
        approved: approvalDecisions[request.tool_call_id] ?? false,
        denial_message: approvalDecisions[request.tool_call_id] ? null : 'Denied in browser UI.',
      }))

      const response = await fetch('/api/chat/resume', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_token: resumeToken, approvals }),
      })
      if (!response.ok) {
        throw new Error(`Resume request failed (${response.status})`)
      }

      setPendingApprovals([])
      await readNdjsonStream(response, handleStreamItem)
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Resume request failed.')
    } finally {
      setIsStreaming(false)
    }
  }

  async function submitElicitation(action) {
    if (!currentElicitation) return

    let content = null
    if (action === 'accept' && currentElicitation.kind === 'form') {
      content = {}
      for (const [fieldName, fieldSchema] of Object.entries(currentElicitation.schema?.properties ?? {})) {
        const rawValue = elicitationValues[fieldName]
        if (fieldSchema.type === 'integer') {
          content[fieldName] = Number.parseInt(rawValue, 10)
        } else if (fieldSchema.type === 'number') {
          content[fieldName] = Number.parseFloat(rawValue)
        } else {
          content[fieldName] = rawValue
        }
      }
    }

    await fetch('/api/mcp/elicitation/respond', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ elicitation_id: currentElicitation.id, action, content }),
    })
    setCurrentElicitation(null)
    setElicitationValues({})
  }

  return (
    <div className="app-shell">
      <main className="chat-column">
        <section className="surface hero">
          <p className="eyebrow">infai browser ui</p>
          <h1>Chat, approve tools, finish OAuth, and open MCP apps in one session.</h1>
        </section>

        <section className="surface messages">
          {statusMessage && <p className="status-message">{statusMessage}</p>}
          <div className="transcript">
            {transcript.map((entry) =>
              entry.kind === 'user' ? (
                <article className="message user-message" key={entry.id}>
                  <p className="message-label">You</p>
                  <div className="bubble user-bubble">{entry.text}</div>
                </article>
              ) : (
                <article className="message assistant-message" key={entry.id}>
                  <p className="message-label">Assistant</p>
                  <div className="bubble assistant-bubble">
                    {entry.thinking && (
                      <details className="thinking-panel">
                        <summary>Thinking</summary>
                        <pre>{entry.thinking}</pre>
                      </details>
                    )}

                    {entry.toolCalls.length > 0 && (
                      <div className="tool-stack">
                        {entry.toolCalls.map((toolCall) => (
                          <section className="tool-card" key={toolCall.toolCallId}>
                            <div className="tool-card-header">
                              <strong>{toolCall.toolName}</strong>
                              <span>{formatToolStatus(toolCall.status)}</span>
                            </div>
                            {toolCall.args != null && (
                              <pre className="tool-payload">{renderValue(toolCall.args)}</pre>
                            )}
                            {toolCall.result != null && (
                              <pre className="tool-payload">{renderValue(toolCall.result)}</pre>
                            )}
                          </section>
                        ))}
                      </div>
                    )}

                    {entry.appLaunches.length > 0 && (
                      <div className="app-launch-list">
                        {entry.appLaunches.map((launch) => (
                          <a className="app-launch-chip" href={launch.url} key={launch.toolCallId} rel="noreferrer" target="_blank">
                            Open {launch.title ?? launch.toolName}
                          </a>
                        ))}
                      </div>
                    )}

                    {entry.response && <div className="assistant-response">{entry.response}</div>}
                    {!entry.thinking && !entry.response && entry.toolCalls.length === 0 && (
                      <p className="assistant-placeholder">Waiting for model output...</p>
                    )}
                  </div>
                </article>
              ),
            )}
          </div>
        </section>

        <form className="surface composer" onSubmit={handlePromptSubmit}>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Ask something or trigger a tool..."
            rows={4}
          />
          <button disabled={isStreaming} type="submit">
            {isStreaming ? 'Streaming...' : 'Send'}
          </button>
        </form>

        {canResume && (
          <section className="surface approvals">
            <h2>Pending tool approvals</h2>
            {pendingApprovals.map((request) => (
              <label className="approval-row" key={request.tool_call_id}>
                <div>
                  <strong>{request.tool_name}</strong>
                  <pre className="tool-payload">{renderValue(request.args)}</pre>
                </div>
                <select
                  value={approvalDecisions[request.tool_call_id] ? 'approve' : 'deny'}
                  onChange={(event) =>
                    setApprovalDecisions((current) => ({
                      ...current,
                      [request.tool_call_id]: event.target.value === 'approve',
                    }))
                  }
                >
                  <option value="approve">Approve</option>
                  <option value="deny">Deny</option>
                </select>
              </label>
            ))}
            <button disabled={isStreaming} onClick={handleResume} type="button">
              Resume run
            </button>
          </section>
        )}
      </main>

      <aside className="side-column">
        {currentAuth && (
          <section className="surface status-card">
            <h2>OAuth in progress</h2>
            <p>Complete the sign-in flow in the opened window.</p>
            <a href={currentAuth.authorization_url} rel="noreferrer" target="_blank">
              Open auth window again
            </a>
          </section>
        )}

        {activeApp && (
          <section className="surface app-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">MCP app</p>
                <h2>{activeApp.title ?? activeApp.tool_name}</h2>
              </div>
              <a href={activeApp.url} rel="noreferrer" target="_blank">
                Open tab
              </a>
            </div>
            <iframe
              sandbox="allow-scripts allow-same-origin allow-forms"
              referrerPolicy="no-referrer"
              src={activeApp.url}
              title={activeApp.title ?? activeApp.tool_name}
            />
          </section>
        )}
      </aside>

      {currentElicitation && (
        <div className="modal-backdrop">
          <section className="surface modal">
            <p className="eyebrow">MCP elicitation</p>
            <h2>{currentElicitation.message}</h2>
            {currentElicitation.kind === 'url' ? (
              <a href={currentElicitation.url} rel="noreferrer" target="_blank">
                Open requested URL
              </a>
            ) : (
              <div className="elicitation-form">
                {Object.entries(currentElicitation.schema?.properties ?? {}).map(([fieldName, fieldSchema]) => (
                  <label key={fieldName}>
                    <span>{fieldName}</span>
                    {fieldSchema.type === 'boolean' ? (
                      <input
                        checked={Boolean(elicitationValues[fieldName])}
                        onChange={(event) =>
                          setElicitationValues((current) => ({
                            ...current,
                            [fieldName]: event.target.checked,
                          }))
                        }
                        type="checkbox"
                      />
                    ) : (
                      <input
                        onChange={(event) =>
                          setElicitationValues((current) => ({
                            ...current,
                            [fieldName]: event.target.value,
                          }))
                        }
                        type="text"
                        value={elicitationValues[fieldName] ?? ''}
                      />
                    )}
                  </label>
                ))}
              </div>
            )}
            <div className="modal-actions">
              <button onClick={() => submitElicitation('accept')} type="button">
                Accept
              </button>
              <button onClick={() => submitElicitation('decline')} type="button">
                Decline
              </button>
              <button onClick={() => submitElicitation('cancel')} type="button">
                Cancel
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
