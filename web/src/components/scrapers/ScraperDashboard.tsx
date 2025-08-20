import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Tab,
  Tabs,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Schedule,
  CheckCircle,
  Error,
  Warning,
  TrendingUp,
  TrendingDown,
  Speed,
  Storage,
} from '@mui/icons-material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { format, formatDistanceToNow } from 'date-fns';

interface ScraperStats {
  total_scrapers: number;
  by_category: Record<string, number>;
  by_platform: Record<string, number>;
  by_status: Record<string, number>;
  by_jurisdiction: Record<string, number>;
  health_metrics: {
    success_rate_24h: number;
    average_runtime: number;
    failed_scrapers: number;
    stale_scrapers: number;
  };
}

interface ActiveRun {
  run_id: string;
  scraper_id: string;
  scraper_name: string;
  start_time: string;
  duration: number;
  status: string;
}

interface ScraperDashboardProps {
  apiUrl: string;
}

const ScraperDashboard: React.FC<ScraperDashboardProps> = ({ apiUrl }) => {
  const [stats, setStats] = useState<ScraperStats | null>(null);
  const [activeRuns, setActiveRuns] = useState<ActiveRun[]>([]);
  const [selectedTab, setSelectedTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedPlatform, setSelectedPlatform] = useState('all');
  const [executionDialogOpen, setExecutionDialogOpen] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      setRefreshing(true);
      
      // Fetch stats
      const statsResponse = await fetch(`${apiUrl}/scrapers/stats`);
      const statsData = await statsResponse.json();
      setStats(statsData);

      // Fetch active runs
      const runsResponse = await fetch(`${apiUrl}/scrapers/runs/active`);
      const runsData = await runsResponse.json();
      setActiveRuns(runsData.runs || []);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching scraper data:', error);
      setLoading(false);
    } finally {
      setRefreshing(false);
    }
  };

  const executeScraper = async (scraperId?: string, category?: string) => {
    try {
      const body = scraperId 
        ? { scraper_ids: [scraperId] }
        : { category };

      const response = await fetch(`${apiUrl}/scrapers/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (response.ok) {
        setExecutionDialogOpen(false);
        fetchData();
      }
    } catch (error) {
      console.error('Error executing scraper:', error);
    }
  };

  const stopScraper = async (runId: string) => {
    try {
      await fetch(`${apiUrl}/scrapers/runs/${runId}/stop`, { method: 'POST' });
      fetchData();
    } catch (error) {
      console.error('Error stopping scraper:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'failed': return 'error';
      case 'inactive': return 'default';
      case 'maintenance': return 'warning';
      default: return 'default';
    }
  };

  const getHealthIcon = (rate: number) => {
    if (rate >= 0.9) return <CheckCircle color="success" />;
    if (rate >= 0.7) return <Warning color="warning" />;
    return <Error color="error" />;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!stats) {
    return <Alert severity="error">Failed to load scraper statistics</Alert>;
  }

  const successRate = stats.health_metrics.success_rate_24h;
  const platformChartData = {
    labels: Object.keys(stats.by_platform),
    datasets: [{
      data: Object.values(stats.by_platform),
      backgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56',
        '#4BC0C0',
        '#9966FF',
        '#FF9F40',
      ],
    }],
  };

  const categoryChartData = {
    labels: Object.keys(stats.by_category).map(cat => cat.replace(/_/g, ' ').toUpperCase()),
    datasets: [{
      label: 'Scrapers by Category',
      data: Object.values(stats.by_category),
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1,
    }],
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Scraper Management Dashboard - 40by6
        </Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={() => setExecutionDialogOpen(true)}
            sx={{ mr: 2 }}
          >
            Execute Scrapers
          </Button>
          <IconButton onClick={fetchData} disabled={refreshing}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Scrapers
              </Typography>
              <Typography variant="h4">
                {stats.total_scrapers.toLocaleString()}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Storage sx={{ mr: 1 }} />
                <Typography variant="body2">
                  {Object.values(stats.by_status).reduce((a, b) => a + b, 0)} configured
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Success Rate (24h)
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="h4">
                  {(successRate * 100).toFixed(1)}%
                </Typography>
                {getHealthIcon(successRate)}
              </Box>
              <LinearProgress
                variant="determinate"
                value={successRate * 100}
                sx={{ mt: 2 }}
                color={successRate >= 0.9 ? 'success' : successRate >= 0.7 ? 'warning' : 'error'}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Runs
              </Typography>
              <Typography variant="h4">
                {activeRuns.length}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Speed sx={{ mr: 1 }} />
                <Typography variant="body2">
                  Avg: {stats.health_metrics.average_runtime.toFixed(1)}s
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Issues
              </Typography>
              <Box>
                <Chip
                  icon={<Error />}
                  label={`${stats.health_metrics.failed_scrapers} Failed`}
                  color="error"
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Chip
                  icon={<Warning />}
                  label={`${stats.health_metrics.stale_scrapers} Stale`}
                  color="warning"
                  size="small"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
          <Tab label="Overview" />
          <Tab label="Active Runs" />
          <Tab label="Analytics" />
          <Tab label="Schedule" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scrapers by Platform
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Doughnut data={platformChartData} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scrapers by Category
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Bar data={categoryChartData} options={{ maintainAspectRatio: false }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Jurisdiction Coverage
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(stats.by_jurisdiction).map(([type, count]) => (
                    <Grid item xs={12} sm={4} key={type}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                        <Typography variant="subtitle1">
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </Typography>
                        <Typography variant="h6">
                          {count}
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 1 && (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Scraper</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Started</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {activeRuns.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No active scraper runs
                  </TableCell>
                </TableRow>
              ) : (
                activeRuns.map((run) => (
                  <TableRow key={run.run_id}>
                    <TableCell>{run.scraper_name}</TableCell>
                    <TableCell>
                      <Chip
                        label={run.status}
                        color={run.status === 'running' ? 'primary' : 'default'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {formatDistanceToNow(new Date(run.start_time), { addSuffix: true })}
                    </TableCell>
                    <TableCell>{run.duration.toFixed(0)}s</TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => stopScraper(run.run_id)}
                        color="error"
                      >
                        <Stop />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Execution Dialog */}
      <Dialog open={executionDialogOpen} onClose={() => setExecutionDialogOpen(false)}>
        <DialogTitle>Execute Scrapers</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              label="Category"
            >
              <MenuItem value="all">All Categories</MenuItem>
              {Object.keys(stats.by_category).map((cat) => (
                <MenuItem key={cat} value={cat}>
                  {cat.replace(/_/g, ' ').toUpperCase()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExecutionDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => executeScraper(undefined, selectedCategory)} variant="contained">
            Execute
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ScraperDashboard;