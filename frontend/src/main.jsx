import React, { useState, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ClinicalProvider } from './context/ClinicalContext'
import App from './App'
import './index.css'

const API_RETRY_MS = 2000
const API_RETRY_ATTEMPTS = 30  // ~60s total - backend may take time to load model on first start

async function waitForApi() {
  for (let i = 0; i < API_RETRY_ATTEMPTS; i++) {
    try {
      const r = await fetch('/api/health')
      if (r.ok) return true
    } catch (_) {}
    await new Promise((resolve) => setTimeout(resolve, API_RETRY_MS))
  }
  return false
}

function ApiGate({ children }) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    waitForApi().then(setReady)
  }, [])
  if (!ready) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', fontFamily: 'system-ui', color: '#666' }}>
        Connecting to GlucoSense…
      </div>
    )
  }
  return children
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ApiGate>
        <ClinicalProvider>
          <App />
        </ClinicalProvider>
      </ApiGate>
    </BrowserRouter>
  </React.StrictMode>,
)
