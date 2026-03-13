import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import GlucoseTrends from './pages/GlucoseTrends'
import InsulinManagement from './pages/InsulinManagement'
import Reports from './pages/Reports'
import Alerts from './pages/Alerts'
import ModelInfo from './pages/ModelInfo'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="glucose-trends" element={<GlucoseTrends />} />
        <Route path="insulin-management" element={<InsulinManagement />} />
        <Route path="reports" element={<Reports />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="model-info" element={<ModelInfo />} />
      </Route>
    </Routes>
  )
}

export default App
