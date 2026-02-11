import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  AppBar,
  Toolbar,
  IconButton,
  TextField,
  MenuItem,
  Grid,
} from '@mui/material';
import { ArrowBack } from '@mui/icons-material';
import api from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PatientHistory = () => {
  const navigate = useNavigate();
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);

  useEffect(() => {
    fetchData();
  }, [days]);

  const fetchData = async () => {
    try {
      const response = await api.get(`/patient/data?days=${days}`);
      setData(response.data.data);
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  const getGlucoseColor = (glucose) => {
    if (!glucose) return 'default';
    if (glucose < 70) return 'error';
    if (glucose > 180) return 'warning';
    return 'success';
  };

  const chartData = data
    .slice()
    .reverse()
    .map((entry) => ({
      date: new Date(entry.timestamp).toLocaleDateString(),
      glucose: entry.glucose_level,
      insulin: entry.insulin_dosage,
      food: entry.food_intake,
    }));

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={() => navigate('/patient/dashboard')}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Data History
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" fontWeight="bold">
                  Your Data History
                </Typography>
                <TextField
                  select
                  value={days}
                  onChange={(e) => setDays(e.target.value)}
                  size="small"
                  sx={{ minWidth: 150 }}
                >
                  <MenuItem value={7}>Last 7 days</MenuItem>
                  <MenuItem value={30}>Last 30 days</MenuItem>
                  <MenuItem value={90}>Last 90 days</MenuItem>
                </TextField>
              </Box>

              {chartData.length > 0 && (
                <Box sx={{ mb: 4 }}>
                  <Typography variant="h6" gutterBottom>
                    Trends Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis yAxisId="left" label={{ value: 'Glucose (mg/dL)', angle: -90 }} />
                      <YAxis yAxisId="right" orientation="right" label={{ value: 'Insulin/Food', angle: 90 }} />
                      <Tooltip />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="glucose" stroke="#1976d2" strokeWidth={2} name="Glucose (mg/dL)" />
                      <Line yAxisId="right" type="monotone" dataKey="insulin" stroke="#dc004e" strokeWidth={2} name="Insulin (units)" />
                      <Line yAxisId="right" type="monotone" dataKey="food" stroke="#2e7d32" strokeWidth={2} name="Food (carbs)" />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              )}

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Date & Time</TableCell>
                      <TableCell align="right">Glucose</TableCell>
                      <TableCell align="right">Insulin</TableCell>
                      <TableCell align="right">Food (carbs)</TableCell>
                      <TableCell align="right">Activity</TableCell>
                      <TableCell>Meal Type</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6} align="center">
                          <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                            No data available for the selected period
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      data.map((entry) => (
                        <TableRow key={entry.id} hover>
                          <TableCell>
                            {new Date(entry.timestamp).toLocaleString()}
                          </TableCell>
                          <TableCell align="right">
                            {entry.glucose_level ? (
                              <Chip
                                label={entry.glucose_level}
                                color={getGlucoseColor(entry.glucose_level)}
                                size="small"
                              />
                            ) : (
                              '-'
                            )}
                          </TableCell>
                          <TableCell align="right">
                            {entry.insulin_dosage || '-'}
                          </TableCell>
                          <TableCell align="right">
                            {entry.food_intake || '-'}
                          </TableCell>
                          <TableCell align="right">
                            {entry.physical_activity ? `${entry.physical_activity} min` : '-'}
                          </TableCell>
                          <TableCell>{entry.meal_type || '-'}</TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default PatientHistory;

