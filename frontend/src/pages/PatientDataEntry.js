import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Grid,
  MenuItem,
  Alert,
  Card,
  CardContent,
  CircularProgress,
  AppBar,
  Toolbar,
  IconButton,
} from '@mui/material';
import { ArrowBack, Save } from '@mui/icons-material';
import api from '../services/api';
import { toast } from 'react-toastify';

const PatientDataEntry = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    glucose_level: '',
    insulin_dosage: '',
    insulin_type_used: '',
    food_intake: '',
    physical_activity: '',
    activity_intensity: 'Medium',
    meal_type: '',
    notes: '',
  });
  const [mlAnalysis, setMlAnalysis] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await api.post('/patient/data', formData);
      setMlAnalysis(response.data.ml_analysis);
      toast.success('Data logged successfully!');
      
      // Reset form after successful submission
      setTimeout(() => {
        setFormData({
          glucose_level: '',
          insulin_dosage: '',
          insulin_type_used: '',
          food_intake: '',
          physical_activity: '',
          activity_intensity: 'Medium',
          meal_type: '',
          notes: '',
        });
        setMlAnalysis(null);
      }, 5000);
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to log data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={() => navigate('/patient/dashboard')}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Log Data
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4 }}>
          <Typography variant="h5" gutterBottom fontWeight="bold">
            Record Your Data
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Enter your glucose levels, insulin dosage, food intake, and activity
          </Typography>

          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Glucose Level (mg/dL)"
                  type="number"
                  name="glucose_level"
                  value={formData.glucose_level}
                  onChange={handleChange}
                  required
                  inputProps={{ min: 0, max: 500 }}
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Insulin Dosage (units)"
                  type="number"
                  name="insulin_dosage"
                  value={formData.insulin_dosage}
                  onChange={handleChange}
                  inputProps={{ min: 0, step: 0.5 }}
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Insulin Type"
                  name="insulin_type_used"
                  value={formData.insulin_type_used}
                  onChange={handleChange}
                  select
                >
                  <MenuItem value="">None</MenuItem>
                  <MenuItem value="Rapid-acting">Rapid-acting</MenuItem>
                  <MenuItem value="Short-acting">Short-acting</MenuItem>
                  <MenuItem value="Intermediate-acting">Intermediate-acting</MenuItem>
                  <MenuItem value="Long-acting">Long-acting</MenuItem>
                </TextField>
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Meal Type"
                  name="meal_type"
                  value={formData.meal_type}
                  onChange={handleChange}
                  select
                >
                  <MenuItem value="">None</MenuItem>
                  <MenuItem value="Breakfast">Breakfast</MenuItem>
                  <MenuItem value="Lunch">Lunch</MenuItem>
                  <MenuItem value="Dinner">Dinner</MenuItem>
                  <MenuItem value="Snack">Snack</MenuItem>
                </TextField>
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Food Intake (carbs in grams)"
                  type="number"
                  name="food_intake"
                  value={formData.food_intake}
                  onChange={handleChange}
                  inputProps={{ min: 0 }}
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Physical Activity (minutes)"
                  type="number"
                  name="physical_activity"
                  value={formData.physical_activity}
                  onChange={handleChange}
                  inputProps={{ min: 0 }}
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Activity Intensity"
                  name="activity_intensity"
                  value={formData.activity_intensity}
                  onChange={handleChange}
                  select
                >
                  <MenuItem value="Low">Low</MenuItem>
                  <MenuItem value="Medium">Medium</MenuItem>
                  <MenuItem value="High">High</MenuItem>
                </TextField>
              </Grid>

              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Notes"
                  name="notes"
                  value={formData.notes}
                  onChange={handleChange}
                  multiline
                  rows={3}
                  placeholder="Any additional notes or observations..."
                />
              </Grid>

              <Grid item xs={12}>
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  startIcon={loading ? <CircularProgress size={20} /> : <Save />}
                  disabled={loading}
                  sx={{ py: 1.5 }}
                >
                  {loading ? 'Saving...' : 'Save Data'}
                </Button>
              </Grid>
            </Grid>
          </form>

          {/* ML Analysis Results */}
          {mlAnalysis && (
            <Card sx={{ mt: 4, bgcolor: 'primary.light', color: 'white' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  AI-Powered Insights
                </Typography>
                <Alert severity="info" sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.2)' }}>
                  {mlAnalysis.safety_note}
                </Alert>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  <strong>Recommendation:</strong> {mlAnalysis.explanation}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Confidence:</strong> {(mlAnalysis.confidence * 100).toFixed(1)}%
                </Typography>
                {mlAnalysis.recommendation && mlAnalysis.recommendation.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Additional Recommendations:
                    </Typography>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {mlAnalysis.recommendation.map((rec, idx) => (
                        <li key={idx}>
                          <Typography variant="body2">{rec}</Typography>
                        </li>
                      ))}
                    </ul>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Paper>
      </Container>
    </Box>
  );
};

export default PatientDataEntry;

