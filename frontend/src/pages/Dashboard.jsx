import { useState, useEffect } from 'react'
import { useClinical } from '../context/ClinicalContext'
import ConfirmDoseModal from '../components/ConfirmDoseModal'
import SuccessToast from '../components/SuccessToast'
import ResourcePanel from '../components/ResourcePanel'

import {
  AGE_MIN, AGE_MAX, GENDER_OPTIONS, FOOD_INTAKE_OPTIONS, PREVIOUS_MEDICATION_OPTIONS,
  GLUCOSE_MIN, GLUCOSE_MAX, BMI_MIN, BMI_MAX, HBA1C_MIN, HBA1C_MAX, WEIGHT_MIN, WEIGHT_MAX,
  IOB_MAX_ML, ANTICIPATED_CARBS_MAX_G, GLUCOSE_TREND_OPTIONS,
  CONFIDENCE_CAUTION_THRESHOLD_PCT, CONFIDENCE_HIGH_PCT, CONFIDENCE_MEDIUM_PCT,
  EXPLANATION_DRIVERS_DISPLAY_LIMIT, CERTAINTY_TOOLTIP,
  MEDICATION_NAME_MAX_LENGTH, DOSE_CONFIRM_DELAY_MS,
} from '../constants'

const API = '/api'

// Core + optional numeric inputs only (most contributing). Other features imputed by pipeline.
const NUMERIC_FIELDS = [
  { key: 'glucose_level', label: 'Glucose (mg/dL) *', required: true },
  { key: 'BMI', label: 'BMI (optional)' },
  { key: 'HbA1c', label: 'HbA1c % (optional)' },
  { key: 'weight', label: 'Weight kg (optional)' },
]
const NUMERIC_OPTIONAL_KEYS = ['physical_activity', 'insulin_sensitivity', 'sleep_hours', 'creatinine', 'family_history']
// Type 1 dosing context (optional): IOB, anticipated carbs, glucose trend
const DOSING_CONTEXT_KEYS = ['iob', 'anticipated_carbs', 'glucose_trend']

const DEFAULT_AGE = '30'

const initialForm = () => {
  const o = { patient_id: '', medication_name: '' }
  o.age = DEFAULT_AGE
  NUMERIC_FIELDS.forEach(({ key }) => { o[key] = '' })
  NUMERIC_OPTIONAL_KEYS.forEach((key) => { o[key] = '' })
  DOSING_CONTEXT_KEYS.forEach((key) => { o[key] = '' })
  o.gender = 'Male'
  o.food_intake = 'Medium'
  o.previous_medications = 'None'
  return o
}

/** Client-side validation; returns list of { field, message }. */
function validateForm(form) {
  const errors = []
  const ageVal = form.age !== '' && form.age != null ? Number(form.age) : null
  if (ageVal === null || ageVal === '') errors.push({ field: 'age', message: 'Age is required.' })
  else if (Number.isNaN(ageVal)) errors.push({ field: 'age', message: 'Age must be a number.' })
  else if (ageVal < AGE_MIN || ageVal > AGE_MAX) errors.push({ field: 'age', message: `Age must be between ${AGE_MIN} and ${AGE_MAX}.` })
  else if (ageVal !== Math.floor(ageVal)) errors.push({ field: 'age', message: 'Age must be a whole number.' })

  const gender = String(form.gender || '').trim()
  if (!gender) errors.push({ field: 'gender', message: 'Gender is required.' })
  else if (!GENDER_OPTIONS.includes(gender)) errors.push({ field: 'gender', message: `Gender must be one of: ${GENDER_OPTIONS.join(', ')}.` })

  const food = String(form.food_intake || '').trim()
  if (!food) errors.push({ field: 'food_intake', message: 'Food intake is required.' })
  else if (!FOOD_INTAKE_OPTIONS.includes(food)) errors.push({ field: 'food_intake', message: `Food intake must be one of: ${FOOD_INTAKE_OPTIONS.join(', ')}.` })

  const prevMed = String(form.previous_medications || '').trim()
  if (!prevMed) errors.push({ field: 'previous_medications', message: 'Previous medication is required.' })
  else if (!PREVIOUS_MEDICATION_OPTIONS.includes(prevMed)) errors.push({ field: 'previous_medications', message: `Previous medication must be one of: ${PREVIOUS_MEDICATION_OPTIONS.join(', ')}.` })
  else if (prevMed === 'Oral') {
    const medName = String(form.medication_name || '').trim()
    if (!medName) errors.push({ field: 'medication_name', message: 'Medication name is required when Previous medication is Oral.' })
  }

  const gl = form.glucose_level !== '' && form.glucose_level != null ? Number(form.glucose_level) : null
  if (gl === null || gl === '') errors.push({ field: 'glucose_level', message: 'Glucose level is required for recommendation.' })
  else if (Number.isNaN(gl)) errors.push({ field: 'glucose_level', message: 'Glucose must be a number.' })
  else if (gl < GLUCOSE_MIN || gl > GLUCOSE_MAX) errors.push({ field: 'glucose_level', message: `Glucose must be between ${GLUCOSE_MIN} and ${GLUCOSE_MAX} mg/dL. Enter a valid reading.` })

  // Optional numeric: if provided, must be in valid medical range (prompt until valid)
  const bmiVal = form.BMI !== '' && form.BMI != null ? Number(form.BMI) : null
  if (bmiVal != null && bmiVal !== '') {
    if (Number.isNaN(bmiVal)) errors.push({ field: 'BMI', message: 'BMI must be a number.' })
    else if (bmiVal < BMI_MIN || bmiVal > BMI_MAX) errors.push({ field: 'BMI', message: `BMI must be between ${BMI_MIN} and ${BMI_MAX} kg/m². Enter a valid value.` })
  }
  const hba1cVal = form.HbA1c !== '' && form.HbA1c != null ? Number(form.HbA1c) : null
  if (hba1cVal != null && hba1cVal !== '') {
    if (Number.isNaN(hba1cVal)) errors.push({ field: 'HbA1c', message: 'HbA1c must be a number.' })
    else if (hba1cVal < HBA1C_MIN || hba1cVal > HBA1C_MAX) errors.push({ field: 'HbA1c', message: `HbA1c must be between ${HBA1C_MIN} and ${HBA1C_MAX}%. Enter a valid value.` })
  }
  const weightVal = form.weight !== '' && form.weight != null ? Number(form.weight) : null
  if (weightVal != null && weightVal !== '') {
    if (Number.isNaN(weightVal)) errors.push({ field: 'weight', message: 'Weight must be a number.' })
    else if (weightVal < WEIGHT_MIN || weightVal > WEIGHT_MAX) errors.push({ field: 'weight', message: `Weight must be between ${WEIGHT_MIN} and ${WEIGHT_MAX} kg. Enter a valid value.` })
  }

  return errors
}

function buildBody(form) {
  const body = {}
  if (form.patient_id) body.patient_id = form.patient_id
  if (form.age !== '' && form.age != null) body.age = Number(form.age)
  if (form.gender) body.gender = form.gender
  if (form.food_intake) body.food_intake = form.food_intake
  if (form.previous_medications) body.previous_medications = form.previous_medications
  if (form.previous_medications === 'Oral' && form.medication_name) body.medication_name = String(form.medication_name).trim()
  NUMERIC_FIELDS.forEach(({ key }) => {
    if (form[key] !== '' && form[key] != null) body[key] = Number(form[key])
  })
  NUMERIC_OPTIONAL_KEYS.forEach((key) => {
    if (form[key] !== '' && form[key] != null) body[key] = key === 'family_history' ? String(form[key]) : Number(form[key])
  })
  // Type 1 dosing context (optional)
  if (form.iob !== '' && form.iob != null && !Number.isNaN(Number(form.iob))) body.iob = Number(form.iob)
  if (form.anticipated_carbs !== '' && form.anticipated_carbs != null && !Number.isNaN(Number(form.anticipated_carbs))) body.anticipated_carbs = Number(form.anticipated_carbs)
  if (form.glucose_trend && String(form.glucose_trend).trim()) body.glucose_trend = String(form.glucose_trend).trim().toLowerCase()
  return body
}

const CLINICAL_LABELS = {
  down: 'Consider decrease',
  up: 'Consider increase',
  steady: 'Maintain current dose',
  no: 'No change',
}

export default function Dashboard() {
  const { setRecentMetrics, setPatient } = useClinical()
  const [form, setForm] = useState(initialForm())
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fieldErrors, setFieldErrors] = useState([]) // [{ field, message }]
  const [confirmDoseOpen, setConfirmDoseOpen] = useState(false)
  const [doseAdministering, setDoseAdministering] = useState(false)
  const [toastShow, setToastShow] = useState(false)
  const [resourcePanelOpen, setResourcePanelOpen] = useState(false)
  const [resourceId, setResourceId] = useState(null)

  useEffect(() => {
    if (result) {
      setRecentMetrics({
        glucose: form.glucose_level ? Number(form.glucose_level) : null,
        carbohydrates: form.food_intake ? null : null,
        activityMinutes: form.physical_activity ? Number(form.physical_activity) : null,
      })
    }
  }, [result, form.glucose_level, form.physical_activity, form.food_intake, setRecentMetrics])

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
    setError(null)
    setFieldErrors((prev) => prev.filter((e) => e.field !== key))
  }

  const getRecommendation = async () => {
    const clientErrors = validateForm(form)
    if (clientErrors.length > 0) {
      setFieldErrors(clientErrors)
      setError('Please fix the errors below.')
      return
    }
    setFieldErrors([])
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(form)),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        if (res.status === 422 && Array.isArray(data.errors)) {
          setFieldErrors(data.errors)
          setError(data.detail || 'Validation failed.')
        } else {
          setError(data.detail || data.message || res.statusText || 'Request failed')
        }
        return
      }
      setResult(data)
    } catch (e) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const clinicalSummary = result
    ? CLINICAL_LABELS[result.predicted_class] || result.recommendation_summary
    : null
  const confidence = result ? Math.round((result.confidence || 0) * 100) : 0
  const isCaution = result && (result.is_high_risk || confidence < CONFIDENCE_CAUTION_THRESHOLD_PCT)
  const certaintyTier = confidence >= CONFIDENCE_HIGH_PCT ? 'High' : confidence >= CONFIDENCE_MEDIUM_PCT ? 'Medium' : 'Low'

  const doseSummary = result
    ? {
        mealBolus: result.dosage_magnitude || 'Per protocol',
        correctionDose: result.dosage_action || '—',
        totalDose: result.recommendation_summary || 'See guidance',
      }
    : null

  const handleConfirmDose = async () => {
    setDoseAdministering(true)
    try {
      await fetch(`${API}/dose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          meal_bolus: doseSummary?.mealBolus,
          correction_dose: doseSummary?.correctionDose,
          total_dose: doseSummary?.totalDose,
        }),
      })
    } catch (_) {}
    await new Promise((r) => setTimeout(r, DOSE_CONFIRM_DELAY_MS))
    setDoseAdministering(false)
    setConfirmDoseOpen(false)
    setToastShow(true)
  }

  const openResource = (id) => {
    setResourceId(id)
    setResourcePanelOpen(true)
  }

  return (
    <div className="dashboard">
      <section className="dashboard-section dashboard-patient-entry">
        <div className="card">
          <h2 className="card-heading">Current assessment</h2>
          <p className="card-description">Enter patient data below. Click Get recommendation to run the current assessment; the result is recorded and appears in trends and reports.</p>
          {fieldErrors.length > 0 && (
            <ul className="form-validation-errors" role="alert" aria-live="polite">
              {fieldErrors.map((err, i) => (
                <li key={i} data-field={err.field}>
                  <strong>{err.field}:</strong> {err.message}
                </li>
              ))}
            </ul>
          )}
          <div className="form-grid">
            <label className="form-field">
              <span className="form-label">Age (years) *</span>
              <input
                type="number"
                min={AGE_MIN}
                max={AGE_MAX}
                step="1"
                value={form.age ?? ''}
                onChange={(e) => handleChange('age', e.target.value)}
                className="form-input"
                aria-invalid={fieldErrors.some((e) => e.field === 'age')}
                aria-describedby={fieldErrors.some((e) => e.field === 'age') ? 'age-error' : undefined}
              />
              {fieldErrors.some((e) => e.field === 'age') && (
                <span id="age-error" className="form-field-error">{fieldErrors.find((e) => e.field === 'age')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Gender *</span>
              <select
                value={form.gender ?? ''}
                onChange={(e) => handleChange('gender', e.target.value)}
                className="form-input form-select"
                aria-invalid={fieldErrors.some((e) => e.field === 'gender')}
              >
                <option value="">Select...</option>
                {GENDER_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {fieldErrors.some((e) => e.field === 'gender') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'gender')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Food intake *</span>
              <select
                value={form.food_intake ?? ''}
                onChange={(e) => handleChange('food_intake', e.target.value)}
                className="form-input form-select"
                aria-invalid={fieldErrors.some((e) => e.field === 'food_intake')}
              >
                <option value="">Select...</option>
                {FOOD_INTAKE_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {fieldErrors.some((e) => e.field === 'food_intake') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'food_intake')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Previous medication *</span>
              <select
                value={form.previous_medications ?? ''}
                onChange={(e) => handleChange('previous_medications', e.target.value)}
                className="form-input form-select"
                aria-invalid={fieldErrors.some((e) => e.field === 'previous_medications')}
              >
                <option value="">Select...</option>
                {PREVIOUS_MEDICATION_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {fieldErrors.some((e) => e.field === 'previous_medications') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'previous_medications')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Glucose (mg/dL) *</span>
              <input
                type="number"
                step="any"
                min={GLUCOSE_MIN}
                max={GLUCOSE_MAX}
                value={form.glucose_level ?? ''}
                onChange={(e) => handleChange('glucose_level', e.target.value)}
                className="form-input"
                aria-invalid={fieldErrors.some((e) => e.field === 'glucose_level')}
                aria-describedby={fieldErrors.some((e) => e.field === 'glucose_level') ? 'glucose-error' : undefined}
              />
              {fieldErrors.some((e) => e.field === 'glucose_level') && (
                <span id="glucose-error" className="form-field-error" role="alert">{fieldErrors.find((e) => e.field === 'glucose_level')?.message}</span>
              )}
            </label>
            {form.previous_medications === 'Oral' && (
              <label className="form-field form-field-full">
                <span className="form-label">Medication name (required for Oral) *</span>
                <input
                  type="text"
                  value={form.medication_name ?? ''}
                  onChange={(e) => handleChange('medication_name', e.target.value)}
                  placeholder="e.g. Metformin"
                  className="form-input"
                  maxLength={MEDICATION_NAME_MAX_LENGTH}
                  aria-invalid={fieldErrors.some((e) => e.field === 'medication_name')}
                />
                {fieldErrors.some((e) => e.field === 'medication_name') && (
                  <span className="form-field-error">{fieldErrors.find((e) => e.field === 'medication_name')?.message}</span>
                )}
              </label>
            )}
            {NUMERIC_FIELDS.filter((f) => f.key !== 'glucose_level').map(({ key, label }) => {
              const min = key === 'BMI' ? BMI_MIN : key === 'HbA1c' ? HBA1C_MIN : key === 'weight' ? WEIGHT_MIN : undefined
              const max = key === 'BMI' ? BMI_MAX : key === 'HbA1c' ? HBA1C_MAX : key === 'weight' ? WEIGHT_MAX : undefined
              return (
                <label key={key} className="form-field">
                  <span className="form-label">{label}</span>
                  <input
                    type="number"
                    step="any"
                    min={min}
                    max={max}
                    value={form[key] ?? ''}
                    onChange={(e) => handleChange(key, e.target.value)}
                    className="form-input"
                    aria-invalid={fieldErrors.some((e) => e.field === key)}
                    aria-describedby={fieldErrors.some((e) => e.field === key) ? `${key}-error` : undefined}
                  />
                  {fieldErrors.some((e) => e.field === key) && (
                    <span id={`${key}-error`} className="form-field-error" role="alert">{fieldErrors.find((e) => e.field === key)?.message}</span>
                  )}
                </label>
              )
            })}
            <div className="form-field form-field-full" style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border)' }}>
              <span className="form-label" style={{ fontWeight: 600 }}>Type 1 dosing context (optional)</span>
              <p className="form-hint" style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>IOB, anticipated carbs, and glucose trend improve safety and context summary.</p>
            </div>
            <label className="form-field">
              <span className="form-label">Insulin on board (IOB, mL)</span>
              <input
                type="number"
                step="0.001"
                min={0}
                max={IOB_MAX_ML}
                value={form.iob ?? ''}
                onChange={(e) => handleChange('iob', e.target.value)}
                placeholder=""
                className="form-input"
                aria-invalid={fieldErrors.some((e) => e.field === 'iob')}
              />
              {fieldErrors.some((e) => e.field === 'iob') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'iob')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Anticipated carbs (g)</span>
              <input
                type="number"
                step="1"
                min={0}
                max={ANTICIPATED_CARBS_MAX_G}
                value={form.anticipated_carbs ?? ''}
                onChange={(e) => handleChange('anticipated_carbs', e.target.value)}
                placeholder=""
                className="form-input"
                aria-invalid={fieldErrors.some((e) => e.field === 'anticipated_carbs')}
              />
              {fieldErrors.some((e) => e.field === 'anticipated_carbs') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'anticipated_carbs')?.message}</span>
              )}
            </label>
            <label className="form-field">
              <span className="form-label">Glucose trend</span>
              <select
                value={form.glucose_trend ?? ''}
                onChange={(e) => handleChange('glucose_trend', e.target.value)}
                className="form-input form-select"
                aria-invalid={fieldErrors.some((e) => e.field === 'glucose_trend')}
              >
                <option value="">Select...</option>
                {GLUCOSE_TREND_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              {fieldErrors.some((e) => e.field === 'glucose_trend') && (
                <span className="form-field-error">{fieldErrors.find((e) => e.field === 'glucose_trend')?.message}</span>
              )}
            </label>
          </div>
          <button type="button" className="btn btn-primary" onClick={getRecommendation} disabled={loading}>
            {loading ? 'Getting recommendation…' : 'Get recommendation'}
          </button>
        </div>
      </section>

      {error && (
        <div className="alert alert-warning" role="alert">{error}</div>
      )}

      {result && (
        <>
          <section className="dashboard-section dashboard-insight-row">
            <div className="card card-ui-recommendation" style={{ gridColumn: '1 / -1' }}>
              <h2 className="card-heading">Current reading</h2>
              <dl className="ui-recommendation-dl">
                <dt>Current Reading</dt>
                <dd>{result.current_reading_display || `${form.glucose_level || '—'} mg/dL`}</dd>
                <dt>Trend</dt>
                <dd>{result.trend_display || '—'}</dd>
                <dt>IOB</dt>
                <dd>{result.iob_display || 'Not provided'}</dd>
                <dt>What the readings suggest</dt>
                <dd className="ui-interpretation">{result.system_interpretation || result.context_summary || result.recommendation_summary}</dd>
                <dt>Recommended Action</dt>
                <dd className="ui-action"><strong>{result.recommended_action || result.recommendation_summary}</strong></dd>
              </dl>
            </div>

            <div className="card card-insight recommendation-card">
              <h2 className="card-heading">Insulin recommendation</h2>
              <div className={`recommendation-insight ${isCaution ? 'recommendation-caution' : ''}`}>
                <div className="recommendation-value">{clinicalSummary}</div>
                <div className="recommendation-confidence" title={CERTAINTY_TOOLTIP}>
                  <span className="confidence-label">
                    Certainty
                    <span className={`confidence-tier confidence-tier--${certaintyTier.toLowerCase()}`}>({certaintyTier})</span>
                  </span>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: `${confidence}%` }} />
                  </div>
                  <span className="confidence-pct">{confidence}%</span>
                  {certaintyTier === 'Low' && (
                    <p className="confidence-note">Multiple options are plausible—use clinical judgment.</p>
                  )}
                </div>
                <p className="recommendation-tooltip" title={result.recommendation_detail}>
                  {result.recommendation_summary}
                </p>
              </div>
            </div>

            <div className="card card-insight dosage-card">
              <h2 className="card-heading">Dosage guidance</h2>
              <div className="dosage-breakdown">
                <div className="dosage-row">
                  <span>Meal bolus</span>
                  <strong>{result.dosage_magnitude || 'Per protocol'}</strong>
                </div>
                <div className="dosage-row">
                  <span>Correction dose</span>
                  <strong>{result.dosage_action || '—'}</strong>
                </div>
                <div className="dosage-row dosage-row-total">
                  <span>Total dose</span>
                  <strong>{result.recommendation_summary || 'See above'}</strong>
                </div>
              </div>
              <button
                type="button"
                className="btn btn-primary btn-administer"
                onClick={() => setConfirmDoseOpen(true)}
              >
                Administer dose
              </button>
            </div>
          </section>

          <section className="dashboard-section dashboard-advice-row">
            <div className="card card-advice">
              <h2 className="card-heading">Adjustment advice</h2>
              <div className={`advice-content ${result.is_high_risk ? 'advice-caution' : ''}`}>
                <p>{result.recommendation_summary}</p>
                {result.context_summary && (
                  <p className="advice-context" style={{ fontWeight: 500, marginTop: '0.5rem', padding: '0.5rem', background: 'var(--surface)', borderRadius: 6 }}>
                    <strong>Context summary:</strong> {result.context_summary}
                  </p>
                )}
                {result.recommendation_detail && <p className="advice-detail">{result.recommendation_detail}</p>}
                {result.is_high_risk && (
                  <div className="advice-flag">
                    <strong>Flag for review:</strong> {result.high_risk_reason || 'System less certain than usual.'}
                  </div>
                )}
              </div>
            </div>

            <div className="card card-factors">
              <h2 className="card-heading">Contributing factors</h2>
              {result.explanation_drivers && result.explanation_drivers.length > 0 ? (
                <ul className="factor-list">
                  {result.explanation_drivers.slice(0, EXPLANATION_DRIVERS_DISPLAY_LIMIT).map((d, i) => (
                    <li key={i}>{d.clinical_sentence || `${d.feature}: ${d.value}`}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-muted">Factors are based on current readings and protocol.</p>
              )}
            </div>
          </section>

          <section className="dashboard-section">
            <h2 className="section-heading">Clinical resources</h2>
            <div className="resources-grid">
              <button type="button" className="resource-card" onClick={() => openResource('hypo')}>
                <span className="resource-card-title">Hypoglycemia protocol</span>
                <span className="resource-card-desc">Recognition and treatment</span>
              </button>
              <button type="button" className="resource-card" onClick={() => openResource('diet')}>
                <span className="resource-card-title">Dietary guidance</span>
                <span className="resource-card-desc">Carb counting and meal planning</span>
              </button>
              <button type="button" className="resource-card" onClick={() => openResource('exercise')}>
                <span className="resource-card-title">Exercise recommendations</span>
                <span className="resource-card-desc">Activity and glucose</span>
              </button>
            </div>
          </section>

          <div className="card card-disclaimer">
            <p className="disclaimer-text">{result.clinical_disclaimer}</p>
          </div>
        </>
      )}

      {!result && !loading && (
        <div className="card card-empty-state">
          <p>Enter patient data above and select <strong>Get recommendation</strong> to see insulin guidance and trends.</p>
        </div>
      )}

      <ConfirmDoseModal
        open={confirmDoseOpen}
        onClose={() => setConfirmDoseOpen(false)}
        onConfirm={handleConfirmDose}
        doseSummary={doseSummary}
        loading={doseAdministering}
      />
      <SuccessToast message="Dose recorded successfully." show={toastShow} onDismiss={() => setToastShow(false)} />
      <ResourcePanel open={resourcePanelOpen} onClose={() => setResourcePanelOpen(false)} resourceId={resourceId} />
    </div>
  )
}
