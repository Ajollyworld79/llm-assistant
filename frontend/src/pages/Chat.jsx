import React from 'react'

const BASE = 'http://127.0.0.1:8000'

export default function Chat(){
  const [query, setQuery] = React.useState('')
  const [results, setResults] = React.useState([])
  const [loading, setLoading] = React.useState(false)

  async function doSearch(){
    if(!query.trim()) return
    setLoading(true)
    const res = await fetch(BASE + '/search', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query, top_k:5})})
    const js = await res.json()
    setResults(js.results || [])
    setLoading(false)
  }

  return (
    <div className="container">
      <div className="banner">Demo mode — results are simulated; no active LLM connection.</div>
      <textarea value={query} onChange={e=>setQuery(e.target.value)} placeholder="Type your query..." />
      <div className="controls">
        <button onClick={doSearch}>Search</button>
        {loading && <span className="spinner" />}
      </div>

      <div className="results">
        {results.length===0 && <p>No results</p>}
        {results.map((r,i)=>(
          <div key={i} className="result">
            <strong>{r.filename}</strong>
            <div className="snippet">{r.chunk_text}</div>
            <div className="score">score: {r.score.toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
