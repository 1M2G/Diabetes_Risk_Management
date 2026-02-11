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
} from '@mui/material';
import { ArrowBack, CheckCircle } from '@mui/icons-material';
import api from '../services/api';
import { toast } from 'react-toastify';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DoctorPatientView = () => {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [summary, setSummary] = useState(null);
  const [data, setData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPatientData();
  }, [patientId]);

  const fetchPatientData = async () => {
    try {
      const [summaryRes, dataRes, alertsRes] = await Promise.all([
        api.get(`/doctor/patients/${patientId}/summary`),
        api.get(`/doctor/patients/${patientId}/data?days=30`),
        api.get(`/doctor/alerts`),
      ]);
      setSummary(summaryRes.data);
      setData(dataRes.data.data);
      setAlerts(alertsRes.data.alerts.filter((a) => a.patient_id === parseInt(patientId)));
    } catch (error) {
      toast.error('Failed to load patient data');
      console.error(error);
    } finally {
      setLoading(false);
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

  const chartData = data
    .slice()
    .reverse()
    .map((entry) => ({
      date: new Date(entry.timestamp).toLocaleDateString(),
      glucose: entry.glucose_level,
    }));

  if (loading) {
    return (
      <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <LinearProgress sx={{ width: '50%' }} />
      </Box>
    );
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={() => navigate('/doctor/dashboard')}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Patient Details
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Summary Stats */}
          {summary && (
            <>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Average Glucose
                    </Typography>
                    <Typography variant="h3" color="primary">
                      {summary.summary?.average_glucose
                        ? Math.round(summary.summary.average_glucose)
                        : 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      mg/dL
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Glucose Range
                    </Typography>
                    {summary.summary?.glucose_range ? (
                      <Typography variant="h5">
                        {Math.round(summary.summary.glucose_range.min)} -{' '}
                        {Math.round(summary.summary.glucose_range.max)}
                      </Typography>
                    ) : (
                      <Typography variant="body2">No data</Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Data Entries
                    </Typography>
                    <Typography variant="h3" color="primary">
                      {summary.summary?.total_entries || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Last 30 days
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}

          {/* Pattern Analysis */}
          {summary?.pattern_analysis && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                  AI Pattern Analysis
                  </Typography>
                  {summary.pattern_analysis.patterns && summary.pattern_analysis.patterns.length > 0 ? (
                    <Box>
                      {summary.pattern_analysis.patterns.map((pattern, idx) => (
                        <Alert key={idx} severity="info" sx={{ mb: 1 }}>
                          {pattern}
                        </Alert>
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No significant patterns detected. Patient data appears stable.
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Alerts */}
          {alerts.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Active Alerts
                  </Typography>
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
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Glucose Chart */}
          {chartData.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Glucose Trends (Last 30 Days)
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis label={{ value: 'Glucose (mg/dL)', angle: -90 }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="glucose" stroke="#1976d2" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Recent Data */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Data Entries
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Date & Time</TableCell>
                        <TableCell>Glucose</TableCell>
                        <TableCell>Insulin</TableCell>
                        <TableCell>Food</TableCell>
                        <TableCell>Activity</TableCell>
                        <TableCell>ML Analysis</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.slice(0, 10).map((entry) => (
                        <TableRow key={entry.id}>
                          <TableCell>
                            {new Date(entry.timestamp).toLocaleString()}
                          </TableCell>
                          <TableCell>{entry.glucose_level || '-'}</TableCell>
                          <TableCell>{entry.insulin_dosage || '-'}</TableCell>
                          <TableCell>{entry.food_intake || '-'}</TableCell>
                          <TableCell>
                            {entry.physical_activity
                              ? `${entry.physical_activity} min`
                              : '-'}
                          </TableCell>
                          <TableCell>
                            {entry.ml_analysis ? (
                              <Chip
                                label={entry.ml_analysis.prediction || 'analyzed'}
                                color="info"
                                size="small"
                              />
                            ) : (
                              '-'
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default DoctorPatientView;

