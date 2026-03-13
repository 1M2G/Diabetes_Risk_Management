/**
 * Domain and UI constants for GlucoSense.
 * Single source of truth for bounds, thresholds, and display values.
 */

// Medical value ranges (must match backend validation)
export const AGE_MIN = 0
export const AGE_MAX = 100
export const GENDER_OPTIONS = ['Male', 'Female']
export const FOOD_INTAKE_OPTIONS = ['Low', 'Medium', 'High']
export const PREVIOUS_MEDICATION_OPTIONS = ['None', 'Insulin', 'Oral']
export const GLUCOSE_MIN = 20
export const GLUCOSE_MAX = 600
export const BMI_MIN = 12
export const BMI_MAX = 70
export const HBA1C_MIN = 4
export const HBA1C_MAX = 20
export const WEIGHT_MIN = 20
export const WEIGHT_MAX = 300

// Type 1 dosing context
// IOB in mL (U-100: 1 mL = 100 units, so 0.5 mL = 50 units)
export const IOB_MIN_ML = 0
export const IOB_MAX_ML = 0.5
export const ANTICIPATED_CARBS_MIN_G = 0
export const ANTICIPATED_CARBS_MAX_G = 500
export const GLUCOSE_TREND_OPTIONS = ['stable', 'rising', 'falling']

// Chart display
export const CHART_REFERENCE_LINE_LOW_MGDL = 70
export const CHART_REFERENCE_LINE_HIGH_MGDL = 180
export const CHART_Y_DOMAIN_MIN = 60
export const CHART_Y_DOMAIN_MAX = 220
export const CHART_HEIGHT_DASHBOARD = 280
export const CHART_HEIGHT_TRENDS = 360

// Recommendation UI
export const CONFIDENCE_CAUTION_THRESHOLD_PCT = 70
export const CONFIDENCE_HIGH_PCT = 60   // Above this = "High" certainty
export const CONFIDENCE_MEDIUM_PCT = 40 // Above this = "Medium"; below = "Low"
export const EXPLANATION_DRIVERS_DISPLAY_LIMIT = 8

// Certainty tooltip (shown on Insulin recommendation card)
export const CERTAINTY_TOOLTIP = 'The model\'s probability for this recommendation. With 4 options (down, up, steady, no), 25% is random chance. Low values mean multiple options are plausible—use clinical judgment.'

// Form / validation
export const MEDICATION_NAME_MAX_LENGTH = 200

// Timing (ms)
export const FETCH_TREND_DEBOUNCE_MS = 300
export const DOSE_CONFIRM_DELAY_MS = 800

// API
export const ALERTS_FETCH_LIMIT = 50
