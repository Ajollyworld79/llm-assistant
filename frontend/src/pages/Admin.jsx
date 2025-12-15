import React from 'react'
const BASE = 'http://127.0.0.1:8000'

export default function Admin(){
  const [files, setFiles] = React.useState([])
  const [docs, setDocs] = React.useState([])
  const [token, setToken] = React.useState(localStorage.getItem('admin_token') || '')
  const [uploading, setUploading] = React.useState(false)
  const [uploadStatus, setUploadStatus] = React.useState('')

  async function refresh(){
    const res = await fetch(BASE + '/documents')
    const js = await res.json()
    setDocs(js)
  }

  async function upload(){
    if(files.length === 0) return alert('choose files')
    setUploading(true)
    setUploadStatus('Uploading...')
    for(let file of files){
      const fd = new FormData(); fd.append('file', file)
      const headers = {}
      if(token) headers['Authorization'] = 'Bearer ' + token
      try {
        const res = await fetch(BASE + '/upload', {method:'POST', body:fd, headers})
        const js = await res.json()
        if(res.status===401) {
          setUploadStatus('Unauthorized - check admin token')
          break
        }
        setUploadStatus(`Uploaded: ${js.filename}`)
      } catch(e) {
        setUploadStatus(`Error uploading ${file.name}`)
      }
    }
    setUploading(false)
    setFiles([])
    refresh()
  }

  async function deleteDoc(docId){
    const headers = {}
    if(token) headers['Authorization'] = 'Bearer ' + token
    const res = await fetch(BASE + `/document/${docId}`, {method:'DELETE', headers})
    if(res.status === 401) return alert('Unauthorized')
    if(res.ok) refresh()
    else alert('Failed to delete')
  }

  function saveToken(){
    localStorage.setItem('admin_token', token)
    alert('Saved token locally')
  }

  function handleDrop(e){
    e.preventDefault()
    setFiles(Array.from(e.dataTransfer.files))
  }

  function handleDragOver(e){
    e.preventDefault()
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
      <div className="drop-zone" onDrop={handleDrop} onDragOver={handleDragOver}>
        <p>Drag and drop files here or click to select</p>
        <input type="file" multiple onChange={(e)=>setFiles(Array.from(e.target.files))} />
      </div>
      {files.length > 0 && <p>Selected: {files.map(f=>f.name).join(', ')}</p>}
      <button onClick={upload} disabled={uploading}>Upload</button>
      {uploading && <span className="spinner" />}
      <p>{uploadStatus}</p>
      <h3>Documents</h3>
      <ul>
        {docs.map(d=> (
          <li key={d.id}>
            {d.filename} — {d.chunks} chunks
            <button onClick={()=>deleteDoc(d.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  )
}
