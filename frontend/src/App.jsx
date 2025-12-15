import React from 'react'
import Chat from './pages/Chat'
import Admin from './pages/Admin'

export default function App(){
  const [view, setView] = React.useState('chat')
  return (
    <div className="app">
      <header>
        <h1>Qdrant Chatbot — Demo Mode</h1>
        <nav>
          <button onClick={()=>setView('chat')} className={view==='chat'? 'active': ''}>Chat</button>
          <button onClick={()=>setView('admin')} className={view==='admin'? 'active': ''}>Admin</button>
        </nav>
      </header>
      <main>
        {view==='chat' ? <Chat /> : <Admin />}
      </main>
    </div>
  )
}
