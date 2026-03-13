import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { ALERTS_FETCH_LIMIT } from '../constants'

const API = '/api'
const PROFILE_STORAGE_KEY = 'glucosense_user_profile'
const REPORTS_DOWNLOADED_KEY = 'glucosense_reports_downloaded_dates'
const REPORTS_DOWNLOAD_TYPE = 'reports_download'

function getRecordDate(record) {
  if (!record?.created_at) return null
  const d = new Date(record.created_at)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function getDatesWithRecords(records) {
  const set = new Set()
  records.forEach((r) => { const d = getRecordDate(r); if (d) set.add(d) })
  return [...set].sort().reverse()
}

function getDownloadedDates() {
  try {
    const raw = localStorage.getItem(REPORTS_DOWNLOADED_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr : []
  } catch { return [] }
}

function formatDateLabel(dateStr) {
  const d = new Date(dateStr + 'T12:00:00')
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
}
const ClinicalContext = createContext(null)

function loadProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY)
    if (raw) {
      const p = JSON.parse(raw)
      return { displayName: p.displayName ?? '', role: p.role ?? '', email: p.email ?? '' }
    }
  } catch (_) {}
  return { displayName: '', role: '', email: '' }
}

export function ClinicalProvider({ children }) {
  const [theme, setTheme] = useState('light')
  const [isSignedIn, setSignedIn] = useState(true)
  const [userProfile, setUserProfileState] = useState(loadProfile)
  const [patient, setPatientState] = useState({
    name: 'Current Patient',
    condition: 'Type 1 Diabetes',
    photoPlaceholder: true,
  })
  const [recentMetrics, setRecentMetrics] = useState({
    glucose: null,
    glucoseUnit: 'mg/dL',
    carbohydrates: null,
    activityMinutes: null,
    timestamp: null,
  })
  const [notifications, setNotifications] = useState([])
  const [alertsPreview, setAlertsPreview] = useState(0)

  const fetchPatientContext = useCallback(async () => {
    try {
      const res = await fetch(`${API}/patient-context`)
      if (!res.ok) return
      const data = await res.json()
      setPatientState((p) => ({
        ...p,
        name: data.name || p.name,
        condition: data.condition || p.condition,
      }))
      setRecentMetrics((prev) => ({
        ...prev,
        glucose: data.glucose ?? prev.glucose,
        carbohydrates: data.carbohydrates ?? prev.carbohydrates,
        activityMinutes: data.activity_minutes ?? prev.activityMinutes,
        timestamp: data.updated_at || prev.timestamp,
      }))
    } catch (_) {}
  }, [])

  const fetchNotifications = useCallback(async () => {
    try {
      const res = await fetch(`${API}/notifications`)
      if (!res.ok) return
      const data = await res.json()
      setNotifications(data.notifications || [])
    } catch (_) {}
  }, [])

  const syncReportsDownloadNotification = useCallback(async () => {
    try {
      const res = await fetch(`${API}/records?limit=100`)
      if (!res.ok) return
      const data = await res.json()
      const records = data.records || []
      const datesWithRecords = getDatesWithRecords(records)
      const downloadedDates = getDownloadedDates()
      const undownloadedDates = datesWithRecords.filter((d) => !downloadedDates.includes(d))
      if (undownloadedDates.length > 0) {
        const label = undownloadedDates.length === 1
          ? formatDateLabel(undownloadedDates[0])
          : `${undownloadedDates.length} days`
        await fetch(`${API}/notifications`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: `Reports from ${label} ready to download. Go to Reports to download before the next session.`,
            type: REPORTS_DOWNLOAD_TYPE,
          }),
        })
      } else {
        await fetch(`${API}/notifications/by-type/${REPORTS_DOWNLOAD_TYPE}`, { method: 'DELETE' })
      }
    } catch (_) {}
  }, [])

  const fetchAlertsPreview = useCallback(async () => {
    try {
      const res = await fetch(`${API}/alerts?limit=${ALERTS_FETCH_LIMIT}&unresolved_only=true`)
      if (!res.ok) return
      const data = await res.json()
      setAlertsPreview((data.alerts || []).length)
    } catch (_) {}
  }, [])

  useEffect(() => {
    const load = async () => {
      await syncReportsDownloadNotification()
      await fetchNotifications()
      fetchPatientContext()
      fetchAlertsPreview()
    }
    load()
  }, [fetchPatientContext, fetchNotifications, fetchAlertsPreview, syncReportsDownloadNotification])

  const updatePatient = useCallback((name, condition) => {
    setPatientState((p) => ({ ...p, name: name || p.name, condition: condition || p.condition }))
  }, [])

  const updateRecentMetrics = useCallback((metrics) => {
    setRecentMetrics((prev) => ({ ...prev, ...metrics, timestamp: metrics.timestamp || new Date().toISOString() }))
  }, [])

  const setUserProfile = useCallback((updates) => {
    setUserProfileState((prev) => {
      const next = { ...prev, ...updates }
      try {
        localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(next))
      } catch (_) {}
      return next
    })
  }, [])

  const clearNotificationBadge = useCallback(async () => {
    try {
      await fetch(`${API}/notifications/read`, { method: 'PATCH' })
      setNotifications((n) => n.map((x) => ({ ...x, unread: false })))
    } catch (_) {
      setNotifications((n) => n.map((x) => ({ ...x, unread: false })))
    }
  }, [])

  const refreshFromApi = useCallback(() => {
    fetchPatientContext()
    fetchNotifications()
    fetchAlertsPreview()
  }, [fetchPatientContext, fetchNotifications, fetchAlertsPreview])

  const value = {
    theme,
    setTheme,
    isSignedIn,
    setSignedIn,
    userProfile,
    setUserProfile,
    patient: { ...patient, photoPlaceholder: true },
    setPatient: updatePatient,
    recentMetrics,
    setRecentMetrics: updateRecentMetrics,
    notifications,
    setNotifications,
    clearNotificationBadge,
    alertsPreview,
    setAlertsPreview: setAlertsPreview,
    refreshFromApi,
  }

  return (
    <ClinicalContext.Provider value={value}>
      {children}
    </ClinicalContext.Provider>
  )
}

export function useClinical() {
  const ctx = useContext(ClinicalContext)
  if (!ctx) throw new Error('useClinical must be used within ClinicalProvider')
  return ctx
}
