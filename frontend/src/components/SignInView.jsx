import { FiActivity } from 'react-icons/fi'
import { useClinical } from '../context/ClinicalContext'

export default function SignInView() {
  const { setSignedIn } = useClinical()

  return (
    <div className="signin-view">
      <div className="signin-card">
        <div className="signin-logo"><FiActivity size={48} /></div>
        <h1 className="signin-title">GlucoSense</h1>
        <p className="signin-subtitle">Clinical Decision Support</p>
        <button type="button" className="signin-btn" onClick={() => setSignedIn(true)}>
          Sign in
        </button>
      </div>
    </div>
  )
}
