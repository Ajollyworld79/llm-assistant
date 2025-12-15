import React from 'react'
const BASE = 'http://127.0.0.1:8000'

export default function Admin(){
  const [file, setFile] = React.useState(null)
  const [docs, setDocs] = React.useState([])
  const [token, setToken] = React.useState(localStorage.getItem('admin_token') || '')

  async function refresh(){
    const res = await fetch(BASE + '/documents')
    const js = await res.json()
    setDocs(js)
  }

  async function upload(){
    if(!file) return alert('choose file')
    const fd = new FormData(); fd.append('file', file)
    const headers = {}
    if(token) headers['Authorization'] = 'Bearer ' + token
    const res = await fetch(BASE + '/upload', {method:'POST', body:fd, headers})
    const js = await res.json()
    if(res.status===401) return alert('Unauthorized - check admin token')
    alert('uploaded: ' + js.filename)
    setFile(null)
    refresh()
  }

  function saveToken(){
    localStorage.setItem('admin_token', token)
    alert('Saved token locally')
  }

  React.useEffect(()=>{refresh()}, [])

  return (
    <div className="container">
      <h2>Upload Documents</h2>
      <div>
        <label>Admin token: </label>
        <input value={token} onChange={e=>setToken(e.target.value)} placeholder="Enter admin token" />
        <button onClick={saveToken}>Save token</button>
      </div>
      <input type="file" onChange={(e)=>setFile(e.target.files[0])} />
      <button onClick={upload}>Upload</button>
      <h3>Documents</h3>
      <ul>
        {docs.map(d=> <li key={d.id}>{d.filename} — {d.chunks} chunks</li>)}
      </ul>
    </div>
  )
}
