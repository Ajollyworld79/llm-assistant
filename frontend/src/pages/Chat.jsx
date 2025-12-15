import React from 'react'

const BASE = 'http://127.0.0.1:8000'

export default function Chat(){
  const [messages, setMessages] = React.useState([])
  const [query, setQuery] = React.useState('')
  const [loading, setLoading] = React.useState(false)
  const [fullDoc, setFullDoc] = React.useState(null)

  async function doSearch(){
    if(!query.trim()) return
    const userMsg = {type: 'user', content: query}
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    const res = await fetch(BASE + '/search', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query, top_k:5})})
    const js = await res.json()
    const botMsg = {type: 'bot', content: 'Here are the search results:', results: js.results || []}
    setMessages(prev => [...prev, botMsg])
    setLoading(false)
    setQuery('')
  }

  async function viewFullDoc(filename){
    const res = await fetch(BASE + `/document/${filename}`)
    if(res.ok){
      const js = await res.json()
      setFullDoc(js)
    }
  }

  function closeFullDoc(){
    setFullDoc(null)
  }

  function handleKeyPress(e){
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault()
      doSearch()
    }
  }

  return (
    <div className="container">
      <div className="banner">Demo mode — results are simulated; no active LLM connection.</div>
      <div className="chat-history">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.type}`}>
            <strong>{msg.type === 'user' ? 'You' : 'Bot'}:</strong> {msg.content}
            {msg.results && (
              <div className="results">
                {msg.results.map((r,j)=>(
                  <div key={j} className="result">
                    <strong>{r.filename}</strong>
                    <div className="snippet">{r.chunk_text}</div>
                    <div className="score">score: {r.score.toFixed(3)}</div>
                    <button onClick={()=>viewFullDoc(r.filename)}>View Full Document</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="message bot"><em>Searching...</em></div>}
      </div>
      <div className="input-area">
        <textarea value={query} onChange={e=>setQuery(e.target.value)} onKeyPress={handleKeyPress} placeholder="Type your query..." />
        <button onClick={doSearch} disabled={loading}>Send</button>
      </div>
      {fullDoc && (
        <div className="modal">
          <div className="modal-content">
            <h3>{fullDoc.filename}</h3>
            <pre>{fullDoc.text}</pre>
            <button onClick={closeFullDoc}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}
