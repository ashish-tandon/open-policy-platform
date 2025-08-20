import React, { useState, useEffect, useMemo } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  IconButton,
  Button,
  Chip,
  Avatar,
  Stack,
  Divider,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  LinearProgress,
  Alert,
  AlertTitle,
  useTheme,
  alpha,
  ToggleButton,
  ToggleButtonGroup,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Warning,
  Error,
  Speed,
  Assessment,
  AttachMoney,
  DataUsage,
  Security,
  CloudQueue,
  People,
  Gavel,
  LocationCity,
  Download,
  Print,
  Email,
  CalendarToday,
  BarChart,
  DonutLarge,
  Timeline,
  BubbleChart,
  ShowChart,
  PieChart,
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
} from 'chart.js';
import { Line, Bar, Doughnut, Radar, Bubble } from 'react-chartjs-2';
import { format, subDays } from 'date-fns';
import CountUp from 'react-countup';
import { motion } from 'framer-motion';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale
);

interface ExecutiveReportingDashboardProps {
  apiUrl: string;
}

interface ExecutiveSummary {
  kpis: {
    systemHealth: number;
    dataQuality: number;
    operationalEfficiency: number;
    costPerRecord: number;
    totalRecords: number;
    activeScrapers: number;
    successRate: number;
    avgResponseTime: number;
  };
  trends: {
    period: string;
    growth: number;
    direction: 'up' | 'down' | 'stable';
  };
  insights: {
    type: 'success' | 'warning' | 'error' | 'info';
    category: string;
    message: string;
    recommendation: string;
  }[];
  costAnalysis: {
    total: number;
    breakdown: Record<string, number>;
    projectedMonthly: number;
    savings: number;
  };
}

const ExecutiveReportingDashboard: React.FC<ExecutiveReportingDashboardProps> = ({ apiUrl }) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');
  const [summary, setSummary] = useState<ExecutiveSummary | null>(null);
  const [selectedView, setSelectedView] = useState<'overview' | 'performance' | 'financial' | 'strategic'>('overview');

  useEffect(() => {
    fetchExecutiveData();
    const interval = setInterval(fetchExecutiveData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchExecutiveData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/analytics/executive-dashboard?range=${timeRange}`);
      const data = await response.json();
      setSummary(data);
    } catch (error) {
      console.error('Error fetching executive data:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = async (format: 'pdf' | 'excel' | 'email') => {
    try {
      const response = await fetch(`${apiUrl}/analytics/export-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format, timeRange }),
      });
      
      if (format === 'email') {
        alert('Report sent to executive team');
      } else {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `executive-report-${format}-${new Date().toISOString()}.${format}`;
        a.click();
      }
    } catch (error) {
      console.error('Error exporting report:', error);
    }
  };

  const getHealthColor = (score: number) => {
    if (score >= 90) return theme.palette.success.main;
    if (score >= 70) return theme.palette.warning.main;
    return theme.palette.error.main;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: value < 1 ? 4 : 2,
    }).format(value);
  };

  if (loading && !summary) {
    return (
      <Box sx={{ p: 3 }}>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={150} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (!summary) {
    return <Alert severity="error">Failed to load executive summary</Alert>;
  }

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: theme.palette.grey[50], pb: 4 }}>
      {/* Header */}
      <Paper
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
          color: 'white',
        }}
      >
        <Grid container alignItems="center" justifyContent="space-between">
          <Grid item>
            <Typography variant="h4" component="h1" fontWeight="bold">
              Executive Dashboard
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.9, mt: 1 }}>
              Open Policy Platform Performance Overview
            </Typography>
          </Grid>
          <Grid item>
            <Stack direction="row" spacing={2} alignItems="center">
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <Select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  sx={{ backgroundColor: alpha(theme.palette.common.white, 0.15), color: 'white' }}
                >
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                  <MenuItem value="90d">Last Quarter</MenuItem>
                </Select>
              </FormControl>
              <IconButton color="inherit" onClick={() => exportReport('pdf')}>
                <Download />
              </IconButton>
              <IconButton color="inherit" onClick={() => exportReport('email')}>
                <Email />
              </IconButton>
            </Stack>
          </Grid>
        </Grid>
      </Paper>

      {/* View Selector */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'center' }}>
        <ToggleButtonGroup
          value={selectedView}
          exclusive
          onChange={(_, value) => value && setSelectedView(value)}
          sx={{ backgroundColor: 'white', boxShadow: 1 }}
        >
          <ToggleButton value="overview">
            <Assessment sx={{ mr: 1 }} />
            Overview
          </ToggleButton>
          <ToggleButton value="performance">
            <Speed sx={{ mr: 1 }} />
            Performance
          </ToggleButton>
          <ToggleButton value="financial">
            <AttachMoney sx={{ mr: 1 }} />
            Financial
          </ToggleButton>
          <ToggleButton value="strategic">
            <Timeline sx={{ mr: 1 }} />
            Strategic
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="overline">
                      System Health
                    </Typography>
                    <Typography variant="h3" component="div" sx={{ color: getHealthColor(summary.kpis.systemHealth) }}>
                      <CountUp end={summary.kpis.systemHealth} duration={1.5} suffix="%" />
                    </Typography>
                    <Chip
                      icon={summary.trends.direction === 'up' ? <TrendingUp /> : <TrendingDown />}
                      label={`${summary.trends.growth > 0 ? '+' : ''}${summary.trends.growth}%`}
                      size="small"
                      color={summary.trends.growth > 0 ? 'success' : 'error'}
                      sx={{ mt: 1 }}
                    />
                  </Box>
                  <Avatar sx={{ bgcolor: alpha(getHealthColor(summary.kpis.systemHealth), 0.1), color: getHealthColor(summary.kpis.systemHealth) }}>
                    <Security />
                  </Avatar>
                </Stack>
              </CardContent>
              <Box sx={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 4, bgcolor: getHealthColor(summary.kpis.systemHealth) }} />
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="overline">
                      Data Quality
                    </Typography>
                    <Typography variant="h3" component="div">
                      <CountUp end={summary.kpis.dataQuality} duration={1.5} decimals={1} suffix="%" />
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                      Above target
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: theme.palette.info.light, color: theme.palette.info.main }}>
                    <DataUsage />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="overline">
                      Total Records
                    </Typography>
                    <Typography variant="h3" component="div">
                      <CountUp end={summary.kpis.totalRecords} duration={1.5} separator="," suffix="M" decimals={2} />
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                      Across all jurisdictions
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: theme.palette.success.light, color: theme.palette.success.main }}>
                    <CloudQueue />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="overline">
                      Cost Efficiency
                    </Typography>
                    <Typography variant="h3" component="div">
                      {formatCurrency(summary.kpis.costPerRecord)}
                    </Typography>
                    <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>
                      {formatCurrency(summary.costAnalysis.savings)} saved
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: theme.palette.warning.light, color: theme.palette.warning.main }}>
                    <AttachMoney />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Main Content Area */}
      {selectedView === 'overview' && (
        <Grid container spacing={3}>
          {/* Executive Insights */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Key Insights & Recommendations
                </Typography>
                <List>
                  {summary.insights.map((insight, index) => (
                    <React.Fragment key={index}>
                      <ListItem alignItems="flex-start">
                        <ListItemAvatar>
                          <Avatar sx={{
                            bgcolor: insight.type === 'success' ? theme.palette.success.light :
                                     insight.type === 'warning' ? theme.palette.warning.light :
                                     insight.type === 'error' ? theme.palette.error.light :
                                     theme.palette.info.light
                          }}>
                            {insight.type === 'success' ? <CheckCircle /> :
                             insight.type === 'warning' ? <Warning /> :
                             insight.type === 'error' ? <Error /> :
                             <Assessment />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={insight.message}
                          secondary={
                            <>
                              <Typography variant="body2" color="textSecondary" gutterBottom>
                                {insight.category}
                              </Typography>
                              <Typography variant="body2" color="primary">
                                Recommendation: {insight.recommendation}
                              </Typography>
                            </>
                          }
                        />
                      </ListItem>
                      {index < summary.insights.length - 1 && <Divider variant="inset" component="li" />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Quick Stats */}
          <Grid item xs={12} md={4}>
            <Stack spacing={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Operational Metrics
                  </Typography>
                  <Stack spacing={2}>
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Success Rate</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {summary.kpis.successRate}%
                        </Typography>
                      </Stack>
                      <LinearProgress variant="determinate" value={summary.kpis.successRate} sx={{ mt: 1 }} />
                    </Box>
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Active Scrapers</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {summary.kpis.activeScrapers}/1732
                        </Typography>
                      </Stack>
                      <LinearProgress 
                        variant="determinate" 
                        value={(summary.kpis.activeScrapers / 1732) * 100} 
                        sx={{ mt: 1 }}
                        color="secondary"
                      />
                    </Box>
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Avg Response Time</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {summary.kpis.avgResponseTime.toFixed(2)}s
                        </Typography>
                      </Stack>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>

              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Coverage by Jurisdiction
                  </Typography>
                  <Box sx={{ height: 200 }}>
                    <Doughnut
                      data={{
                        labels: ['Federal', 'Provincial', 'Municipal'],
                        datasets: [{
                          data: [15, 35, 50],
                          backgroundColor: [
                            theme.palette.primary.main,
                            theme.palette.secondary.main,
                            theme.palette.info.main,
                          ],
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'bottom' as const,
                          },
                        },
                      }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          {/* Performance Trends */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Performance Trends
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Line
                    data={{
                      labels: Array.from({ length: 30 }, (_, i) => format(subDays(new Date(), 29 - i), 'MMM dd')),
                      datasets: [
                        {
                          label: 'System Health',
                          data: Array.from({ length: 30 }, () => Math.random() * 10 + 85),
                          borderColor: theme.palette.primary.main,
                          backgroundColor: alpha(theme.palette.primary.main, 0.1),
                          tension: 0.4,
                          fill: true,
                        },
                        {
                          label: 'Data Quality',
                          data: Array.from({ length: 30 }, () => Math.random() * 5 + 90),
                          borderColor: theme.palette.secondary.main,
                          backgroundColor: alpha(theme.palette.secondary.main, 0.1),
                          tension: 0.4,
                          fill: true,
                        },
                        {
                          label: 'Success Rate',
                          data: Array.from({ length: 30 }, () => Math.random() * 8 + 87),
                          borderColor: theme.palette.success.main,
                          backgroundColor: alpha(theme.palette.success.main, 0.1),
                          tension: 0.4,
                          fill: true,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      interaction: {
                        mode: 'index' as const,
                        intersect: false,
                      },
                      plugins: {
                        legend: {
                          position: 'top' as const,
                        },
                        tooltip: {
                          callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`,
                          },
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: false,
                          min: 75,
                          max: 100,
                          ticks: {
                            callback: (value) => `${value}%`,
                          },
                        },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedView === 'financial' && (
        <Grid container spacing={3}>
          {/* Cost Analysis */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Cost Breakdown
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Doughnut
                    data={{
                      labels: Object.keys(summary.costAnalysis.breakdown),
                      datasets: [{
                        data: Object.values(summary.costAnalysis.breakdown),
                        backgroundColor: [
                          theme.palette.primary.main,
                          theme.palette.secondary.main,
                          theme.palette.warning.main,
                          theme.palette.info.main,
                          theme.palette.success.main,
                        ],
                      }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'right' as const,
                        },
                        tooltip: {
                          callbacks: {
                            label: (context) => {
                              const label = context.label || '';
                              const value = context.parsed;
                              return `${label}: ${formatCurrency(value)}`;
                            },
                          },
                        },
                      },
                    }}
                  />
                </Box>
                <Divider sx={{ my: 2 }} />
                <Stack spacing={1}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="body2">Total Monthly Cost</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatCurrency(summary.costAnalysis.total)}
                    </Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="body2">Projected (Annual)</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatCurrency(summary.costAnalysis.projectedMonthly * 12)}
                    </Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="body2" color="success.main">YoY Savings</Typography>
                    <Typography variant="body2" fontWeight="bold" color="success.main">
                      {formatCurrency(summary.costAnalysis.savings)}
                    </Typography>
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          </Grid>

          {/* ROI Analysis */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Return on Investment
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Bar
                    data={{
                      labels: ['Q1', 'Q2', 'Q3', 'Q4'],
                      datasets: [
                        {
                          label: 'Cost',
                          data: [85000, 82000, 78000, 75000],
                          backgroundColor: theme.palette.error.main,
                        },
                        {
                          label: 'Value Generated',
                          data: [120000, 135000, 145000, 160000],
                          backgroundColor: theme.palette.success.main,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'top' as const,
                        },
                        tooltip: {
                          callbacks: {
                            label: (context) => `${context.dataset.label}: ${formatCurrency(context.parsed.y)}`,
                          },
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          ticks: {
                            callback: (value) => formatCurrency(value as number),
                          },
                        },
                      },
                    }}
                  />
                </Box>
                <Alert severity="success" sx={{ mt: 2 }}>
                  <AlertTitle>ROI: 213%</AlertTitle>
                  Platform generating 2.13x return on investment
                </Alert>
              </CardContent>
            </Card>
          </Grid>

          {/* Cost Efficiency Metrics */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Efficiency Metrics Comparison
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Radar
                    data={{
                      labels: [
                        'Cost per Record',
                        'Processing Speed',
                        'Data Quality',
                        'System Uptime',
                        'Resource Utilization',
                        'Error Rate',
                      ],
                      datasets: [
                        {
                          label: 'Current Performance',
                          data: [95, 88, 92, 98, 85, 94],
                          borderColor: theme.palette.primary.main,
                          backgroundColor: alpha(theme.palette.primary.main, 0.2),
                        },
                        {
                          label: 'Industry Benchmark',
                          data: [75, 70, 80, 90, 70, 85],
                          borderColor: theme.palette.grey[400],
                          backgroundColor: alpha(theme.palette.grey[400], 0.1),
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        r: {
                          beginAtZero: true,
                          max: 100,
                        },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Bottom Action Bar */}
      <Paper
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          p: 2,
          backgroundColor: 'background.paper',
          borderTop: 1,
          borderColor: 'divider',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography variant="body2" color="textSecondary">
          Last updated: {format(new Date(), 'PPpp')}
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button variant="outlined" startIcon={<Print />} onClick={() => window.print()}>
            Print Report
          </Button>
          <Button variant="contained" startIcon={<CalendarToday />} onClick={() => exportReport('pdf')}>
            Schedule Report
          </Button>
        </Stack>
      </Paper>
    </Box>
  );
};

export default ExecutiveReportingDashboard;