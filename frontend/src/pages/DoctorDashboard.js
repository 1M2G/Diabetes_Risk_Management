import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  AppBar,
  Toolbar,
  IconButton,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
} from '@mui/material';
import {
  People,
  Warning,
  TrendingUp,
  Logout,
  Visibility,
  Notifications,
} from '@mui/icons-material';
import { useAuth } from '../services/AuthContext';
import api from '../services/api';

const DoctorDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [patientsRes, alertsRes] = await Promise.all([
        api.get('/doctor/patients'),
        api.get('/doctor/alerts?status=active&limit=10'),
      ]);
      setPatients(patientsRes.data.patients);
      setAlerts(alertsRes.data.alerts);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'error',
      high: 'warning',
      medium: 'info',
      low: 'success',
    };
    return colors[severity] || 'default';
  };

  const criticalAlerts = alerts.filter((a) => a.severity === 'critical');
  const highAlerts = alerts.filter((a) => a.severity === 'high');

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Doctor Dashboard
          </Typography>
          <IconButton color="inherit" onClick={() => navigate('/doctor/alerts')}>
            <Notifications />
          </IconButton>
          <IconButton color="inherit" onClick={handleLogout}>
            <Logout />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        {loading ? (
          <LinearProgress />
        ) : (
          <Grid container spacing={3}>
            {/* Welcome Card */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Welcome, Dr. {user?.last_name}!
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monitor your patients and review AI-powered insights
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Stats Cards */}
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <People sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                  <Typography variant="h4">{patients.length}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Patients
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Warning sx={{ fontSize: 40, color: 'error.main', mb: 1 }} />
                  <Typography variant="h4">{criticalAlerts.length}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Critical Alerts
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Warning sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                  <Typography variant="h4">{highAlerts.length}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    High Priority
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <TrendingUp sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                  <Typography variant="h4">{alerts.length}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Active Alerts
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Critical Alerts */}
            {criticalAlerts.length > 0 && (
              <Grid item xs={12}>
                <Card sx={{ border: '2px solid', borderColor: 'error.main' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Warning color="error" sx={{ mr: 1 }} />
                      <Typography variant="h6">Critical Alerts - Immediate Attention Required</Typography>
                    </Box>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Patient</TableCell>
                            <TableCell>Alert Type</TableCell>
                            <TableCell>Message</TableCell>
                            <TableCell>Time</TableCell>
                            <TableCell>Action</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {criticalAlerts.map((alert) => (
                            <TableRow key={alert.id}>
                              <TableCell>Patient #{alert.patient_id}</TableCell>
                              <TableCell>
                                <Chip label={alert.alert_type} color="error" size="small" />
                              </TableCell>
                              <TableCell>{alert.message}</TableCell>
                              <TableCell>
                                {new Date(alert.created_at).toLocaleString()}
                              </TableCell>
                              <TableCell>
                                <Button
                                  size="small"
                                  variant="outlined"
                                  onClick={() => navigate(`/doctor/patient/${alert.patient_id}`)}
                                >
                                  View
                                </Button>
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

            {/* Patients List */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Your Patients
                  </Typography>
                  {patients.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                      No patients assigned yet
                    </Typography>
                  ) : (
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Patient Name</TableCell>
                            <TableCell>Email</TableCell>
                            <TableCell>Avg Glucose</TableCell>
                            <TableCell>Active Alerts</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Action</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {patients.map((patient) => (
                            <TableRow key={patient.id} hover>
                              <TableCell>
                                {patient.first_name} {patient.last_name}
                              </TableCell>
                              <TableCell>{patient.email}</TableCell>
                              <TableCell>
                                {patient.summary?.average_glucose
                                  ? Math.round(patient.summary.average_glucose)
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={patient.active_alerts_count || 0}
                                  color={patient.active_alerts_count > 0 ? 'warning' : 'default'}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={patient.summary?.recent_trend || 'stable'}
                                  color={
                                    patient.summary?.recent_trend === 'increasing'
                                      ? 'warning'
                                      : patient.summary?.recent_trend === 'decreasing'
                                      ? 'info'
                                      : 'success'
                                  }
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Button
                                  size="small"
                                  variant="contained"
                                  startIcon={<Visibility />}
                                  onClick={() => navigate(`/doctor/patient/${patient.id}`)}
                                >
                                  View Details
                                </Button>
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
          </Grid>
        )}
      </Container>
    </Box>
  );
};

export default DoctorDashboard;

