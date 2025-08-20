import React, { useState, useEffect, useMemo } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  Tabs,
  Tab,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Button,
  IconButton,
  LinearProgress,
  Alert,
  Autocomplete,
  TextField,
  Stack,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  People,
  Gavel,
  HowToVote,
  Event,
  LocationCity,
  Flag,
  Assessment,
  Timeline,
  PieChart,
  BarChart,
  Map as MapIcon,
  FilterList,
  Download,
  Refresh,
  DateRange,
  AccountBalance,
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
} from 'chart.js';
import { Line, Bar, Doughnut, Radar, Scatter } from 'react-chartjs-2';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider, DatePicker } from '@mui/x-date-pickers';
import { format, subDays, startOfMonth, endOfMonth } from 'date-fns';
import ReactWordcloud from 'react-wordcloud';
import { MapContainer, TileLayer, Marker, Popup, Choropleth } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement,
  RadialLinearScale
);

interface DataVisualizationDashboardProps {
  apiUrl: string;
}

interface JurisdictionData {
  code: string;
  name: string;
  type: 'federal' | 'provincial' | 'municipal';
  stats: {
    representatives: number;
    bills: number;
    votes: number;
    committees: number;
    events: number;
  };
  trends: {
    date: string;
    activity: number;
  }[];
}

interface LegislativeActivity {
  date: string;
  bills_introduced: number;
  votes_held: number;
  committee_meetings: number;
  total_activity: number;
}

interface RepresentativeStats {
  total: number;
  by_party: Record<string, number>;
  by_role: Record<string, number>;
  by_jurisdiction: Record<string, number>;
  most_active: {
    name: string;
    bills_sponsored: number;
    votes_participated: number;
    committees: number;
  }[];
}

interface BillStats {
  total: number;
  by_status: Record<string, number>;
  by_category: Record<string, number>;
  recent_bills: {
    id: string;
    title: string;
    sponsor: string;
    status: string;
    introduced_date: string;
  }[];
  passage_rate: number;
}

const DataVisualizationDashboard: React.FC<DataVisualizationDashboardProps> = ({ apiUrl }) => {
  const theme = useTheme();
  const [selectedTab, setSelectedTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState({
    start: subDays(new Date(), 30),
    end: new Date(),
  });
  const [selectedJurisdiction, setSelectedJurisdiction] = useState<string>('all');
  const [selectedView, setSelectedView] = useState<'overview' | 'detailed'>('overview');
  
  // Data states
  const [jurisdictionData, setJurisdictionData] = useState<JurisdictionData[]>([]);
  const [legislativeActivity, setLegislativeActivity] = useState<LegislativeActivity[]>([]);
  const [representativeStats, setRepresentativeStats] = useState<RepresentativeStats | null>(null);
  const [billStats, setBillStats] = useState<BillStats | null>(null);
  const [wordCloudData, setWordCloudData] = useState<{ text: string; value: number }[]>([]);

  useEffect(() => {
    fetchAllData();
  }, [dateRange, selectedJurisdiction]);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      // Fetch multiple data sources in parallel
      const [jurisdictions, activity, reps, bills, keywords] = await Promise.all([
        fetchJurisdictionData(),
        fetchLegislativeActivity(),
        fetchRepresentativeStats(),
        fetchBillStats(),
        fetchKeywordData(),
      ]);

      setJurisdictionData(jurisdictions);
      setLegislativeActivity(activity);
      setRepresentativeStats(reps);
      setBillStats(bills);
      setWordCloudData(keywords);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchJurisdictionData = async (): Promise<JurisdictionData[]> => {
    // Simulated data - replace with actual API call
    return [
      {
        code: 'ca',
        name: 'Canada',
        type: 'federal',
        stats: {
          representatives: 338,
          bills: 1245,
          votes: 3421,
          committees: 28,
          events: 567,
        },
        trends: Array.from({ length: 30 }, (_, i) => ({
          date: format(subDays(new Date(), 29 - i), 'yyyy-MM-dd'),
          activity: Math.floor(Math.random() * 100) + 50,
        })),
      },
      {
        code: 'on',
        name: 'Ontario',
        type: 'provincial',
        stats: {
          representatives: 124,
          bills: 567,
          votes: 1234,
          committees: 15,
          events: 234,
        },
        trends: Array.from({ length: 30 }, (_, i) => ({
          date: format(subDays(new Date(), 29 - i), 'yyyy-MM-dd'),
          activity: Math.floor(Math.random() * 50) + 25,
        })),
      },
      {
        code: 'toronto',
        name: 'Toronto',
        type: 'municipal',
        stats: {
          representatives: 25,
          bills: 234,
          votes: 567,
          committees: 8,
          events: 123,
        },
        trends: Array.from({ length: 30 }, (_, i) => ({
          date: format(subDays(new Date(), 29 - i), 'yyyy-MM-dd'),
          activity: Math.floor(Math.random() * 30) + 10,
        })),
      },
    ];
  };

  const fetchLegislativeActivity = async (): Promise<LegislativeActivity[]> => {
    // Simulated data
    return Array.from({ length: 30 }, (_, i) => ({
      date: format(subDays(new Date(), 29 - i), 'yyyy-MM-dd'),
      bills_introduced: Math.floor(Math.random() * 10) + 1,
      votes_held: Math.floor(Math.random() * 20) + 5,
      committee_meetings: Math.floor(Math.random() * 15) + 3,
      total_activity: 0,
    })).map(day => ({
      ...day,
      total_activity: day.bills_introduced + day.votes_held + day.committee_meetings,
    }));
  };

  const fetchRepresentativeStats = async (): Promise<RepresentativeStats> => {
    // Simulated data
    return {
      total: 1234,
      by_party: {
        'Liberal': 456,
        'Conservative': 389,
        'NDP': 234,
        'Bloc Québécois': 89,
        'Green': 45,
        'Independent': 21,
      },
      by_role: {
        'MP': 338,
        'MPP': 124,
        'MLA': 87,
        'Councillor': 685,
      },
      by_jurisdiction: {
        'Federal': 338,
        'Provincial': 456,
        'Municipal': 440,
      },
      most_active: [
        { name: 'John Doe', bills_sponsored: 23, votes_participated: 456, committees: 8 },
        { name: 'Jane Smith', bills_sponsored: 19, votes_participated: 432, committees: 6 },
        { name: 'Bob Johnson', bills_sponsored: 17, votes_participated: 398, committees: 7 },
      ],
    };
  };

  const fetchBillStats = async (): Promise<BillStats> => {
    // Simulated data
    return {
      total: 2456,
      by_status: {
        'First Reading': 234,
        'Second Reading': 189,
        'Committee': 156,
        'Third Reading': 98,
        'Royal Assent': 456,
        'Defeated': 123,
      },
      by_category: {
        'Finance': 345,
        'Health': 289,
        'Environment': 234,
        'Education': 198,
        'Infrastructure': 167,
        'Justice': 156,
        'Other': 1067,
      },
      recent_bills: [
        {
          id: 'C-123',
          title: 'Act to Amend the Income Tax Act',
          sponsor: 'Minister of Finance',
          status: 'Second Reading',
          introduced_date: '2024-01-15',
        },
        {
          id: 'C-124',
          title: 'Environmental Protection Act',
          sponsor: 'Minister of Environment',
          status: 'Committee',
          introduced_date: '2024-01-12',
        },
      ],
      passage_rate: 0.73,
    };
  };

  const fetchKeywordData = async (): Promise<{ text: string; value: number }[]> => {
    // Simulated word cloud data from bill titles and debates
    const keywords = [
      { text: 'Climate', value: 89 },
      { text: 'Healthcare', value: 76 },
      { text: 'Economy', value: 72 },
      { text: 'Education', value: 65 },
      { text: 'Infrastructure', value: 61 },
      { text: 'Housing', value: 58 },
      { text: 'Environment', value: 54 },
      { text: 'Technology', value: 49 },
      { text: 'Immigration', value: 45 },
      { text: 'Budget', value: 42 },
      { text: 'Tax', value: 40 },
      { text: 'Employment', value: 38 },
      { text: 'Security', value: 35 },
      { text: 'Transportation', value: 33 },
      { text: 'Energy', value: 31 },
    ];
    return keywords;
  };

  // Memoized chart data
  const activityChartData = useMemo(() => ({
    labels: legislativeActivity.map(d => format(new Date(d.date), 'MMM dd')),
    datasets: [
      {
        label: 'Bills Introduced',
        data: legislativeActivity.map(d => d.bills_introduced),
        borderColor: theme.palette.primary.main,
        backgroundColor: theme.palette.primary.light,
        tension: 0.4,
      },
      {
        label: 'Votes Held',
        data: legislativeActivity.map(d => d.votes_held),
        borderColor: theme.palette.secondary.main,
        backgroundColor: theme.palette.secondary.light,
        tension: 0.4,
      },
      {
        label: 'Committee Meetings',
        data: legislativeActivity.map(d => d.committee_meetings),
        borderColor: theme.palette.warning.main,
        backgroundColor: theme.palette.warning.light,
        tension: 0.4,
      },
    ],
  }), [legislativeActivity, theme]);

  const representativeChartData = useMemo(() => ({
    labels: Object.keys(representativeStats?.by_party || {}),
    datasets: [{
      data: Object.values(representativeStats?.by_party || {}),
      backgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56',
        '#4BC0C0',
        '#9966FF',
        '#FF9F40',
      ],
    }],
  }), [representativeStats]);

  const billStatusChartData = useMemo(() => ({
    labels: Object.keys(billStats?.by_status || {}),
    datasets: [{
      label: 'Bills by Status',
      data: Object.values(billStats?.by_status || {}),
      backgroundColor: theme.palette.primary.main,
      borderColor: theme.palette.primary.dark,
      borderWidth: 1,
    }],
  }), [billStats, theme]);

  const jurisdictionComparisonData = useMemo(() => ({
    labels: jurisdictionData.map(j => j.name),
    datasets: [
      {
        label: 'Representatives',
        data: jurisdictionData.map(j => j.stats.representatives),
        backgroundColor: '#FF6384',
      },
      {
        label: 'Bills',
        data: jurisdictionData.map(j => j.stats.bills),
        backgroundColor: '#36A2EB',
      },
      {
        label: 'Votes',
        data: jurisdictionData.map(j => j.stats.votes),
        backgroundColor: '#FFCE56',
      },
    ],
  }), [jurisdictionData]);

  const exportData = () => {
    const data = {
      dateRange,
      jurisdictionData,
      legislativeActivity,
      representativeStats,
      billStats,
      exportDate: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `policy-data-export-${format(new Date(), 'yyyy-MM-dd')}.json`;
    a.click();
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3, backgroundColor: theme.palette.grey[50], minHeight: '100vh' }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h4" component="h1" gutterBottom>
              📊 Policy Data Visualization Dashboard
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Comprehensive insights from 1700+ government data sources
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Stack direction="row" spacing={2} justifyContent="flex-end">
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <DatePicker
                  label="Start Date"
                  value={dateRange.start}
                  onChange={(date) => date && setDateRange({ ...dateRange, start: date })}
                  renderInput={(params) => <TextField {...params} size="small" />}
                />
                <DatePicker
                  label="End Date"
                  value={dateRange.end}
                  onChange={(date) => date && setDateRange({ ...dateRange, end: date })}
                  renderInput={(params) => <TextField {...params} size="small" />}
                />
              </LocalizationProvider>
              <Button
                variant="outlined"
                startIcon={<Download />}
                onClick={exportData}
              >
                Export
              </Button>
              <IconButton onClick={fetchAllData}>
                <Refresh />
              </IconButton>
            </Stack>
          </Grid>
        </Grid>
      </Paper>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Representatives
                  </Typography>
                  <Typography variant="h4">
                    {representativeStats?.total.toLocaleString()}
                  </Typography>
                  <Chip
                    label="+12% from last month"
                    size="small"
                    color="success"
                    icon={<TrendingUp />}
                  />
                </Box>
                <People fontSize="large" color="primary" />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Bills
                  </Typography>
                  <Typography variant="h4">
                    {billStats?.total.toLocaleString()}
                  </Typography>
                  <Chip
                    label={`${(billStats?.passage_rate || 0) * 100}% passage rate`}
                    size="small"
                    color="info"
                  />
                </Box>
                <Gavel fontSize="large" color="secondary" />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Jurisdictions
                  </Typography>
                  <Typography variant="h4">
                    {jurisdictionData.length}
                  </Typography>
                  <Chip
                    label="All levels covered"
                    size="small"
                    color="success"
                  />
                </Box>
                <LocationCity fontSize="large" color="warning" />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Monthly Activity
                  </Typography>
                  <Typography variant="h4">
                    {legislativeActivity.reduce((sum, day) => sum + day.total_activity, 0).toLocaleString()}
                  </Typography>
                  <Chip
                    label="-5% from last month"
                    size="small"
                    color="error"
                    icon={<TrendingDown />}
                  />
                </Box>
                <Assessment fontSize="large" color="error" />
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedTab}
          onChange={(_, v) => setSelectedTab(v)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Overview" icon={<PieChart />} iconPosition="start" />
          <Tab label="Legislative Activity" icon={<Timeline />} iconPosition="start" />
          <Tab label="Representatives" icon={<People />} iconPosition="start" />
          <Tab label="Bills & Votes" icon={<Gavel />} iconPosition="start" />
          <Tab label="Geographic View" icon={<MapIcon />} iconPosition="start" />
          <Tab label="Trends & Insights" icon={<Assessment />} iconPosition="start" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          {/* Legislative Activity Timeline */}
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Legislative Activity Timeline
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Line
                    data={activityChartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Word Cloud */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Trending Topics
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ReactWordcloud
                    words={wordCloudData}
                    options={{
                      rotations: 2,
                      rotationAngles: [-90, 0],
                      fontSizes: [20, 60],
                      colors: [
                        theme.palette.primary.main,
                        theme.palette.secondary.main,
                        theme.palette.warning.main,
                        theme.palette.error.main,
                        theme.palette.info.main,
                      ],
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Jurisdiction Comparison */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Jurisdiction Comparison
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Bar
                    data={jurisdictionComparisonData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                      scales: {
                        x: { stacked: false },
                        y: { stacked: false },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 1 && (
        <Grid container spacing={3}>
          {/* Detailed Activity Chart */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    Legislative Activity Details
                  </Typography>
                  <FormControl size="small" sx={{ minWidth: 200 }}>
                    <InputLabel>Jurisdiction</InputLabel>
                    <Select
                      value={selectedJurisdiction}
                      onChange={(e) => setSelectedJurisdiction(e.target.value)}
                      label="Jurisdiction"
                    >
                      <MenuItem value="all">All Jurisdictions</MenuItem>
                      {jurisdictionData.map(j => (
                        <MenuItem key={j.code} value={j.code}>{j.name}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Stack>
                <Box sx={{ height: 500 }}>
                  <Line
                    data={activityChartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      interaction: {
                        mode: 'index' as const,
                        intersect: false,
                      },
                      plugins: {
                        legend: { position: 'top' as const },
                        tooltip: {
                          callbacks: {
                            footer: (tooltipItems) => {
                              const sum = tooltipItems.reduce((a, b) => a + b.parsed.y, 0);
                              return 'Total: ' + sum;
                            },
                          },
                        },
                      },
                      scales: {
                        y: {
                          stacked: true,
                        },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Activity Breakdown */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Activity by Type (Last 30 Days)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Doughnut
                    data={{
                      labels: ['Bills', 'Votes', 'Committee Meetings', 'Events'],
                      datasets: [{
                        data: [
                          legislativeActivity.reduce((sum, d) => sum + d.bills_introduced, 0),
                          legislativeActivity.reduce((sum, d) => sum + d.votes_held, 0),
                          legislativeActivity.reduce((sum, d) => sum + d.committee_meetings, 0),
                          123, // Events (simulated)
                        ],
                        backgroundColor: [
                          theme.palette.primary.main,
                          theme.palette.secondary.main,
                          theme.palette.warning.main,
                          theme.palette.info.main,
                        ],
                      }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'right' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Peak Activity Times */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Peak Activity Times
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Radar
                    data={{
                      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                      datasets: [{
                        label: 'Legislative Activity',
                        data: [85, 92, 95, 88, 76, 20, 15],
                        borderColor: theme.palette.primary.main,
                        backgroundColor: theme.palette.primary.light + '40',
                      }],
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

      {selectedTab === 2 && (
        <Grid container spacing={3}>
          {/* Representatives by Party */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Representatives by Party
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Doughnut
                    data={representativeChartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'right' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Most Active Representatives */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Most Active Representatives
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {representativeStats?.most_active.map((rep, index) => (
                    <Box key={index} sx={{ mb: 2 }}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Box>
                          <Typography variant="subtitle1">{rep.name}</Typography>
                          <Stack direction="row" spacing={1}>
                            <Chip
                              label={`${rep.bills_sponsored} bills`}
                              size="small"
                              color="primary"
                            />
                            <Chip
                              label={`${rep.votes_participated} votes`}
                              size="small"
                              color="secondary"
                            />
                            <Chip
                              label={`${rep.committees} committees`}
                              size="small"
                              color="info"
                            />
                          </Stack>
                        </Box>
                        <Box sx={{ minWidth: 100 }}>
                          <LinearProgress
                            variant="determinate"
                            value={(rep.bills_sponsored / 30) * 100}
                            sx={{ height: 8, borderRadius: 4 }}
                          />
                        </Box>
                      </Stack>
                      <Divider sx={{ mt: 2 }} />
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Representatives by Role */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Representatives by Role & Jurisdiction
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Bar
                    data={{
                      labels: Object.keys(representativeStats?.by_role || {}),
                      datasets: [{
                        label: 'Number of Representatives',
                        data: Object.values(representativeStats?.by_role || {}),
                        backgroundColor: [
                          theme.palette.primary.main,
                          theme.palette.secondary.main,
                          theme.palette.warning.main,
                          theme.palette.info.main,
                        ],
                      }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 3 && (
        <Grid container spacing={3}>
          {/* Bill Status Distribution */}
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Bill Status Distribution
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Bar
                    data={billStatusChartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      indexAxis: 'y' as const,
                      plugins: {
                        legend: { display: false },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Bills */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Bills
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {billStats?.recent_bills.map((bill) => (
                    <Box key={bill.id} sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary">
                        {bill.id}
                      </Typography>
                      <Typography variant="body2">
                        {bill.title}
                      </Typography>
                      <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                        <Chip
                          label={bill.status}
                          size="small"
                          color={bill.status.includes('Reading') ? 'info' : 'default'}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {bill.sponsor}
                        </Typography>
                      </Stack>
                      <Divider sx={{ mt: 2 }} />
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Bill Categories */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Bills by Category
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Bar
                    data={{
                      labels: Object.keys(billStats?.by_category || {}),
                      datasets: [{
                        label: 'Number of Bills',
                        data: Object.values(billStats?.by_category || {}),
                        backgroundColor: theme.palette.primary.main,
                      }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 4 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Geographic Distribution of Legislative Activity
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  Interactive map showing legislative activity across Canada
                </Alert>
                <Box sx={{ height: 600, position: 'relative' }}>
                  {/* Map would go here - using placeholder for now */}
                  <Box
                    sx={{
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: theme.palette.grey[200],
                      borderRadius: 1,
                    }}
                  >
                    <Stack alignItems="center" spacing={2}>
                      <MapIcon sx={{ fontSize: 64, color: theme.palette.grey[400] }} />
                      <Typography variant="h6" color="text.secondary">
                        Interactive Map View
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Showing activity density by region
                      </Typography>
                    </Stack>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 5 && (
        <Grid container spacing={3}>
          {/* Trend Analysis */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Legislative Trends & Predictions
                </Typography>
                <Box sx={{ height: 400 }}>
                  <Scatter
                    data={{
                      datasets: [
                        {
                          label: 'Bills vs Passage Rate',
                          data: Array.from({ length: 50 }, () => ({
                            x: Math.random() * 100,
                            y: Math.random() * 100,
                          })),
                          backgroundColor: theme.palette.primary.main,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: 'Number of Bills Introduced',
                          },
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Passage Rate (%)',
                          },
                        },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Key Insights */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Key Insights & Recommendations
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, backgroundColor: theme.palette.primary.light + '20' }}>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <TrendingUp color="primary" />
                        <Box>
                          <Typography variant="subtitle2">
                            Increasing Activity
                          </Typography>
                          <Typography variant="body2">
                            Legislative activity up 23% in environmental bills
                          </Typography>
                        </Box>
                      </Stack>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, backgroundColor: theme.palette.warning.light + '20' }}>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <Flag color="warning" />
                        <Box>
                          <Typography variant="subtitle2">
                            Regional Disparities
                          </Typography>
                          <Typography variant="body2">
                            Western provinces show 40% higher activity
                          </Typography>
                        </Box>
                      </Stack>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, backgroundColor: theme.palette.success.light + '20' }}>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <AccountBalance color="success" />
                        <Box>
                          <Typography variant="subtitle2">
                            Committee Efficiency
                          </Typography>
                          <Typography variant="body2">
                            Average review time reduced by 15 days
                          </Typography>
                        </Box>
                      </Stack>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default DataVisualizationDashboard;