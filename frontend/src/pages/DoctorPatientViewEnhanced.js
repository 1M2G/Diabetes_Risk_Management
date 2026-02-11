import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  AppBar,
  Toolbar,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  LinearProgress,
  Tabs,
  Tab,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ArrowBack,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Warning,
  Info,
  ExpandMore,
  Refresh,
} from '@mui/icons-material';
import api from '../services/api';
import { toast } from 'react-toastify';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const DoctorPatientViewEnhanced = () => {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [summary, setSummary] = useState(null);
  const [data, setData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchPatientData();
    
    // Auto-refresh every 30 seconds if enabled
    let interval;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchPatientData(true);
      }, 30000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [patientId, autoRefresh]);

  const fetchPatientData = async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);

    try {
      const [summaryRes, dataRes, alertsRes] = await Promise.all([
        api.get(`/doctor/patients/${patientId}/summary?days=14`),
        api.get(`/doctor/patients/${patientId}/data?days=14&limit=200`),
        api.get(`/doctor/alerts`),
      ]);
      setSummary(summaryRes.data);
      setData(dataRes.data.data);
      setAlerts(alertsRes.data.alerts.filter((a) => a.patient_id === parseInt(patientId)));
    } catch (error) {
      if (!silent) toast.error('Failed to load patient data');
      console.error(error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleAcknowledgeAlert = async (alertId) => {
    try {
      await api.post(`/doctor/alerts/${alertId}/acknowledge`);
      toast.success('Alert acknowledged');
      fetchPatientData();
    } catch (error) {
      toast.error('Failed to acknowledge alert');
    }
  };

  // Prepare chart data
  const chartData = data
    .slice()
    .reverse()
    .map((entry) => ({
      date: new Date(entry.timestamp).toLocaleDateString(),
      time: new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      glucose: entry.glucose_level,
      insulin: entry.insulin_dosage,
      food: entry.food_intake,
    }));

  // Time in Range pie chart data
  const tirData = summary?.pattern_analysis?.metrics ? [
    {
      name: 'In Range (70-180)',
      value: (summary.pattern_analysis.metrics.time_in_range || 0) * 100,
      color: '#4caf50',
    },
    {
      name: 'Below Range (<70)',
      value: (summary.pattern_analysis.metrics.time_below_range || 0) * 100,
      color: '#f44336',
    },
    {
      name: 'Above Range (>180)',
      value: (summary.pattern_analysis.metrics.time_above_range || 0) * 100,
      color: '#ff9800',
    },
  ] : [];

  if (loading) {
    return (
      <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  const metrics = summary?.pattern_analysis?.metrics || {};
  const assessment = summary?.pattern_analysis?.assessment || {};

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={() => navigate('/doctor/dashboard')}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Patient Details - Type 1 Diabetes Management
          </Typography>
          <IconButton color="inherit" onClick={() => fetchPatientData()} disabled={refreshing}>
            <Refresh />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {refreshing && (
          <LinearProgress sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1300 }} />
        )}

        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 3 }}>
          <Tab label="Overview" />
          <Tab label="Type 1 Metrics" />
          <Tab label="Trends & Patterns" />
          <Tab label="ML Insights" />
          <Tab label="Alerts" />
        </Tabs>

        {/* Overview Tab */}
        {tabValue === 0 && (
          <Grid container spacing={3}>
            {/* Key Metrics Cards */}
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="text.secondary">
                    Average Glucose
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {summary?.summary?.average_glucose
                      ? Math.round(summary.summary.average_glucose)
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    mg/dL
                  </Typography>
                  {summary?.summary?.average_glucose && (
                    <Chip
                      label={
                        summary.summary.average_glucose >= 70 && summary.summary.average_glucose <= 180
                          ? 'In Range'
                          : 'Out of Range'
                      }
                      color={
                        summary.summary.average_glucose >= 70 && summary.summary.average_glucose <= 180
                          ? 'success'
                          : 'warning'
                      }
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="text.secondary">
                    Time in Range
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {metrics.time_in_range
                      ? `${(metrics.time_in_range * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Target: ≥70%
                  </Typography>
                  {metrics.time_in_range && (
                    <Chip
                      label={metrics.time_in_range >= 0.7 ? 'On Target' : 'Below Target'}
                      color={metrics.time_in_range >= 0.7 ? 'success' : 'warning'}
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="text.secondary">
                    Glucose Variability
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {metrics.glucose_variability_cv
                      ? `${metrics.glucose_variability_cv.toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Target: <36%
                  </Typography>
                  {metrics.glucose_variability_cv && (
                    <Chip
                      label={metrics.glucose_variability_cv < 36 ? 'Stable' : 'Variable'}
                      color={metrics.glucose_variability_cv < 36 ? 'success' : 'warning'}
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="text.secondary">
                    Stability Score
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {metrics.stability_score
                      ? `${metrics.stability_score.toFixed(0)}/100`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Overall Control
                  </Typography>
                  {metrics.stability_score && (
                    <Chip
                      label={
                        metrics.stability_score >= 80
                          ? 'Excellent'
                          : metrics.stability_score >= 60
                          ? 'Good'
                          : 'Needs Improvement'
                      }
                      color={
                        metrics.stability_score >= 80
                          ? 'success'
                          : metrics.stability_score >= 60
                          ? 'info'
                          : 'warning'
                      }
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Control Assessment */}
            {assessment.level && (
              <Grid item xs={12}>
                <Alert
                  severity={
                    assessment.level === 'excellent'
                      ? 'success'
                      : assessment.level === 'good'
                      ? 'info'
                      : assessment.level === 'needs_improvement'
                      ? 'warning'
                      : 'error'
                  }
                  icon={<Info />}
                >
                  <Typography variant="h6" gutterBottom>
                    Control Assessment: {assessment.level.toUpperCase()}
                  </Typography>
                  <Typography variant="body2">{assessment.message}</Typography>
                  {assessment.recommendations && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Recommendations:
                      </Typography>
                      <ul style={{ margin: 0, paddingLeft: 20 }}>
                        {assessment.recommendations.map((rec, idx) => (
                          <li key={idx}>
                            <Typography variant="body2">{rec}</Typography>
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Alert>
              </Grid>
            )}

            {/* Recent Glucose Chart */}
            {chartData.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Glucose Trends (Last 14 Days)
                    </Typography>
                    <ResponsiveContainer width="100%" height={400}>
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="colorGlucose" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#1976d2" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#1976d2" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis
                          label={{ value: 'Glucose (mg/dL)', angle: -90, position: 'insideLeft' }}
                          domain={[0, 300]}
                        />
                        <Tooltip />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="glucose"
                          stroke="#1976d2"
                          fillOpacity={1}
                          fill="url(#colorGlucose)"
                          name="Glucose (mg/dL)"
                        />
                        <Line
                          type="monotone"
                          dataKey="glucose"
                          stroke="#1976d2"
                          strokeWidth={2}
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        )}

        {/* Type 1 Metrics Tab */}
        {tabValue === 1 && (
          <Grid container spacing={3}>
            {/* Time in Range Pie Chart */}
            {tirData.length > 0 && (
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Time in Range Distribution
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={tirData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {tirData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Detailed Metrics Table */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Type 1 Diabetes Metrics
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell><strong>Time in Range</strong></TableCell>
                          <TableCell align="right">
                            {metrics.time_in_range
                              ? `${(metrics.time_in_range * 100).toFixed(1)}%`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metrics.time_in_range >= 0.7 ? '✓' : '✗'}
                              color={metrics.time_in_range >= 0.7 ? 'success' : 'warning'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Time Below Range</strong></TableCell>
                          <TableCell align="right">
                            {metrics.time_below_range
                              ? `${(metrics.time_below_range * 100).toFixed(1)}%`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metrics.time_below_range < 0.04 ? '✓' : '✗'}
                              color={metrics.time_below_range < 0.04 ? 'success' : 'error'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Time Above Range</strong></TableCell>
                          <TableCell align="right">
                            {metrics.time_above_range
                              ? `${(metrics.time_above_range * 100).toFixed(1)}%`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metrics.time_above_range < 0.25 ? '✓' : '✗'}
                              color={metrics.time_above_range < 0.25 ? 'success' : 'warning'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Mean Glucose</strong></TableCell>
                          <TableCell align="right">
                            {metrics.mean_glucose
                              ? `${metrics.mean_glucose.toFixed(1)} mg/dL`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metrics.mean_glucose >= 154 ? 'Target' : 'Below'}
                              color="info"
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Glucose Variability (CV)</strong></TableCell>
                          <TableCell align="right">
                            {metrics.glucose_variability_cv
                              ? `${metrics.glucose_variability_cv.toFixed(1)}%`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metrics.glucose_variability_cv < 36 ? 'Stable' : 'Variable'}
                              color={metrics.glucose_variability_cv < 36 ? 'success' : 'warning'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>GMI (Est. A1C)</strong></TableCell>
                          <TableCell align="right">
                            {metrics.glucose_management_indicator
                              ? `${metrics.glucose_management_indicator.toFixed(1)}%`
                              : 'N/A'}
                          </TableCell>
                          <TableCell>-</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Pattern Analysis */}
            {summary?.pattern_analysis?.patterns && summary.pattern_analysis.patterns.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Identified Patterns
                    </Typography>
                    {summary.pattern_analysis.patterns.map((pattern, idx) => (
                      <Alert key={idx} severity="info" sx={{ mb: 1 }}>
                        {pattern}
                      </Alert>
                    ))}
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        )}

        {/* Trends & Patterns Tab */}
        {tabValue === 2 && (
          <Grid container spacing={3}>
            {chartData.length > 0 && (
              <>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Glucose & Insulin Trends
                      </Typography>
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis yAxisId="left" label={{ value: 'Glucose (mg/dL)', angle: -90 }} />
                          <YAxis yAxisId="right" orientation="right" label={{ value: 'Insulin/Food', angle: 90 }} />
                          <Tooltip />
                          <Legend />
                          <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="glucose"
                            stroke="#1976d2"
                            strokeWidth={2}
                            name="Glucose (mg/dL)"
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="insulin"
                            stroke="#dc004e"
                            strokeWidth={2}
                            name="Insulin (units)"
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="food"
                            stroke="#2e7d32"
                            strokeWidth={2}
                            name="Food (carbs)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </Grid>
              </>
            )}
          </Grid>
        )}

        {/* ML Insights Tab */}
        {tabValue === 3 && (
          <Grid container spacing={3}>
            {data.slice(0, 10).map((entry) => (
              <Grid item xs={12} key={entry.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">
                        {new Date(entry.timestamp).toLocaleString()}
                      </Typography>
                      <Chip
                        label={`Glucose: ${entry.glucose_level || 'N/A'} mg/dL`}
                        color={
                          entry.glucose_level >= 70 && entry.glucose_level <= 180
                            ? 'success'
                            : entry.glucose_level < 70
                            ? 'error'
                            : 'warning'
                        }
                      />
                    </Box>
                    {entry.ml_analysis && (
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMore />}>
                          <Typography>ML Analysis & Recommendations</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Box>
                            <Typography variant="subtitle2" gutterBottom>
                              Recommended Dose: {entry.ml_analysis.recommended_dose} units
                            </Typography>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Confidence: {(entry.ml_analysis.confidence * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" sx={{ mt: 2 }}>
                              {entry.ml_analysis.explanation}
                            </Typography>
                            {entry.ml_analysis.reasoning_steps && (
                              <Box sx={{ mt: 2 }}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Reasoning Steps:
                                </Typography>
                                {entry.ml_analysis.reasoning_steps.map((step, idx) => (
                                  <Alert key={idx} severity="info" sx={{ mb: 1 }}>
                                    <Typography variant="body2">
                                      <strong>Step {step.step}:</strong> {step.description}
                                    </Typography>
                                  </Alert>
                                ))}
                              </Box>
                            )}
                          </Box>
                        </AccordionDetails>
                      </Accordion>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {/* Alerts Tab */}
        {tabValue === 4 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Alerts
                </Typography>
                {alerts.length === 0 ? (
                  <Alert severity="success">No active alerts</Alert>
                ) : (
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Type</TableCell>
                          <TableCell>Severity</TableCell>
                          <TableCell>Message</TableCell>
                          <TableCell>Time</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Action</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {alerts.map((alert) => (
                          <TableRow key={alert.id}>
                            <TableCell>{alert.alert_type}</TableCell>
                            <TableCell>
                              <Chip
                                label={alert.severity}
                                color={
                                  alert.severity === 'critical'
                                    ? 'error'
                                    : alert.severity === 'high'
                                    ? 'warning'
                                    : 'info'
                                }
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{alert.message}</TableCell>
                            <TableCell>
                              {new Date(alert.created_at).toLocaleString()}
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={alert.status}
                                color={alert.status === 'active' ? 'warning' : 'success'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              {alert.status === 'active' && (
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<CheckCircle />}
                                  onClick={() => handleAcknowledgeAlert(alert.id)}
                                >
                                  Acknowledge
                                </Button>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Container>
    </Box>
  );
};

export default DoctorPatientViewEnhanced;

