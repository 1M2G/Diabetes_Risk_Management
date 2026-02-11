import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  AppBar,
  Toolbar,
  IconButton,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  AddCircleOutline,
  History,
  TrendingUp,
  Warning,
  Logout,
  Person,
} from '@mui/icons-material';
import { useAuth } from '../services/AuthContext';
import api from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PatientDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [summary, setSummary] = useState(null);
  const [recentData, setRecentData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [summaryRes, dataRes] = await Promise.all([
        api.get('/patient/summary'),
        api.get('/patient/data?limit=7'),
      ]);
      setSummary(summaryRes.data.summary);
      setRecentData(dataRes.data.data);
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

  const getGlucoseStatus = (glucose) => {
    if (!glucose) return { color: 'default', label: 'No Data' };
    if (glucose < 70) return { color: 'error', label: 'Low' };
    if (glucose > 180) return { color: 'warning', label: 'High' };
    return { color: 'success', label: 'Normal' };
  };

  const chartData = recentData
    .slice()
    .reverse()
    .map((entry) => ({
      date: new Date(entry.timestamp).toLocaleDateString(),
      glucose: entry.glucose_level,
    }));

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Patient Dashboard
          </Typography>
          <IconButton color="inherit" onClick={() => navigate('/patient/data-entry')}>
            <Person />
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
                    Welcome, {user?.first_name}!
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monitor your glucose levels and manage your insulin dosage
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Quick Actions */}
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/patient/data-entry')}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <AddCircleOutline sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6">Log Data</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Record glucose, food, and activity
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/patient/history')}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <History sx={{ fontSize: 60, color: 'secondary.main', mb: 2 }} />
                  <Typography variant="h6">View History</Typography>
                  <Typography variant="body2" color="text.secondary">
                    See your data trends
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <TrendingUp sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
                  <Typography variant="h6">Insights</Typography>
                  <Typography variant="body2" color="text.secondary">
                    AI-powered recommendations
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

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
                        {summary.average_glucose ? Math.round(summary.average_glucose) : 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        mg/dL
                      </Typography>
                      {summary.average_glucose && (
                        <Chip
                          label={getGlucoseStatus(summary.average_glucose).label}
                          color={getGlucoseStatus(summary.average_glucose).color}
                          size="small"
                          sx={{ mt: 1 }}
                        />
                      )}
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Glucose Range
                      </Typography>
                      {summary.glucose_range ? (
                        <>
                          <Typography variant="h5">
                            {Math.round(summary.glucose_range.min)} - {Math.round(summary.glucose_range.max)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            mg/dL
                          </Typography>
                        </>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No data available
                        </Typography>
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
                        {summary.total_entries}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Last 30 days
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </>
            )}

            {/* Recent Glucose Chart */}
            {chartData.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Recent Glucose Trends
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis label={{ value: 'Glucose (mg/dL)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="glucose" stroke="#1976d2" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Recent Alerts */}
            {summary?.recent_alerts && summary.recent_alerts.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Warning color="warning" sx={{ mr: 1 }} />
                      <Typography variant="h6">Recent Alerts</Typography>
                    </Box>
                    {summary.recent_alerts.map((alert) => (
                      <Chip
                        key={alert.id}
                        label={alert.message}
                        color={alert.severity === 'critical' ? 'error' : 'warning'}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        )}
      </Container>
    </Box>
  );
};

export default PatientDashboard;

