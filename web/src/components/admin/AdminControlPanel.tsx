"""
Administrative Control Panel - 40by6
Comprehensive admin interface for managing the entire MCP Stack
"""

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Paper, Grid, Typography, Card, CardContent, CardActions,
  Button, TextField, Switch, Select, MenuItem, Slider, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Dialog, DialogTitle, DialogContent, DialogActions, DialogContentText,
  Tabs, Tab, AppBar, Toolbar, IconButton, Badge, Avatar,
  List, ListItem, ListItemText, ListItemIcon, ListItemSecondaryAction,
  Accordion, AccordionSummary, AccordionDetails, AccordionActions,
  FormControl, FormControlLabel, FormLabel, RadioGroup, Radio,
  Stepper, Step, StepLabel, StepContent, StepButton,
  Alert, AlertTitle, Snackbar, LinearProgress, CircularProgress,
  Drawer, Divider, Menu, Tooltip, Collapse, Pagination,
  InputAdornment, ToggleButton, ToggleButtonGroup, Autocomplete,
  Stack, Breadcrumbs, Link, Skeleton, SpeedDial, SpeedDialAction,
  Container, CssBaseline, ThemeProvider, createTheme
} from '@mui/material';
import {
  Dashboard, Settings, Security, Storage, Speed, People,
  Notifications, Code, CloudUpload, Schedule, Assessment,
  Warning, CheckCircle, Error, Info, ExpandMore, ExpandLess,
  Add, Edit, Delete, Save, Cancel, Refresh, Download,
  Upload, Search, FilterList, Sort, MoreVert, Close,
  PlayArrow, Pause, Stop, RestartAlt, Build, BugReport,
  Timeline, Memory, NetworkCheck, DataUsage, Policy,
  Lock, LockOpen, VpnKey, Shield, VerifiedUser,
  AdminPanelSettings, SupervisorAccount, ManageAccounts,
  Analytics, Insights, TrendingUp, ShowChart, BarChart,
  CloudQueue, CloudDownload, CloudSync, CloudOff,
  IntegrationInstructions, Api, Webhook, Extension,
  Inventory, Category, Label, LocalOffer, Sell,
  MonetizationOn, CreditCard, AccountBalance, Receipt,
  NotificationsActive, NotificationsOff, MarkEmailRead,
  PhoneInTalk, Sms, Chat, Forum, QuestionAnswer
} from '@mui/icons-material';
import { alpha, styled } from '@mui/material/styles';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';
import { DataGrid, GridColDef, GridToolbar } from '@mui/x-data-grid';
import { DateTimePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { TreeView, TreeItem } from '@mui/x-tree-view';
import { LineChart, Line, AreaChart, Area, BarChart as RechartsBarChart,
  Bar, PieChart, Pie, RadarChart, Radar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend,
  ResponsiveContainer, Cell, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm, Controller } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import axios from 'axios';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { formatDistanceToNow, format, parseISO } from 'date-fns';
import { useDropzone } from 'react-dropzone';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor,
  useSensor, useSensors } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates,
  verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Custom styled components
const StyledBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    backgroundColor: '#44b700',
    color: '#44b700',
    boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
    '&::after': {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      borderRadius: '50%',
      animation: 'ripple 1.2s infinite ease-in-out',
      border: '1px solid currentColor',
      content: '""',
    },
  },
  '@keyframes ripple': {
    '0%': {
      transform: 'scale(.8)',
      opacity: 1,
    },
    '100%': {
      transform: 'scale(2.4)',
      opacity: 0,
    },
  },
}));

const StyledTreeItem = styled(TreeItem)(({ theme }) => ({
  [`& .MuiTreeItem-iconContainer`]: {
    '& .close': {
      opacity: 0.3,
    },
  },
  [`& .MuiTreeItem-group`]: {
    marginLeft: 15,
    paddingLeft: 18,
    borderLeft: `1px dashed ${alpha(theme.palette.text.primary, 0.4)}`,
  },
}));

// Theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

// API client
const apiClient = axios.create({
  baseURL: process.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth interceptor
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('admin_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Types
interface SystemStatus {
  overall: 'healthy' | 'degraded' | 'critical';
  services: {
    [key: string]: {
      status: 'up' | 'down' | 'degraded';
      uptime: number;
      cpu: number;
      memory: number;
      errors: number;
    };
  };
  metrics: {
    totalScrapers: number;
    activeScrapers: number;
    totalRuns: number;
    successRate: number;
    dataProcessed: number;
    averageResponseTime: number;
  };
}

interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'operator' | 'viewer';
  lastLogin: string;
  status: 'active' | 'inactive' | 'suspended';
  permissions: string[];
}

interface Configuration {
  id: string;
  category: string;
  key: string;
  value: any;
  type: 'string' | 'number' | 'boolean' | 'json' | 'list';
  description: string;
  editable: boolean;
  requiresRestart: boolean;
}

interface AuditLog {
  id: string;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  details: any;
  ip: string;
  userAgent: string;
  result: 'success' | 'failure';
}

interface BackupJob {
  id: string;
  name: string;
  schedule: string;
  lastRun: string;
  nextRun: string;
  status: 'active' | 'paused' | 'failed';
  size: number;
  duration: number;
  retention: number;
}

interface Integration {
  id: string;
  name: string;
  type: string;
  status: 'connected' | 'disconnected' | 'error';
  lastSync: string;
  config: any;
  stats: {
    totalSynced: number;
    failedSyncs: number;
    averageLatency: number;
  };
}

// Admin Control Panel Component
export const AdminControlPanel: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const queryClient = useQueryClient();
  
  // State management
  const [currentTab, setCurrentTab] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [selectedService, setSelectedService] = useState<string | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<Configuration | null>(null);
  const [userDialogOpen, setUserDialogOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [confirmDialog, setConfirmDialog] = useState({
    open: false,
    title: '',
    message: '',
    onConfirm: () => {},
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterOptions, setFilterOptions] = useState({
    status: 'all',
    role: 'all',
    timeRange: '24h',
  });
  const [notifications, setNotifications] = useState<any[]>([]);
  const [notificationMenuAnchor, setNotificationMenuAnchor] = useState<null | HTMLElement>(null);
  
  // Fetch system status
  const { data: systemStatus, isLoading: statusLoading } = useQuery(
    'systemStatus',
    async () => {
      const response = await apiClient.get('/admin/system/status');
      return response.data;
    },
    { refetchInterval: 5000 }
  );
  
  // Fetch users
  const { data: users, isLoading: usersLoading } = useQuery(
    ['users', searchTerm, filterOptions],
    async () => {
      const response = await apiClient.get('/admin/users', {
        params: { search: searchTerm, ...filterOptions },
      });
      return response.data;
    }
  );
  
  // Fetch configurations
  const { data: configurations, isLoading: configsLoading } = useQuery(
    'configurations',
    async () => {
      const response = await apiClient.get('/admin/configurations');
      return response.data;
    }
  );
  
  // Fetch audit logs
  const { data: auditLogs, isLoading: auditLoading } = useQuery(
    ['auditLogs', filterOptions.timeRange],
    async () => {
      const response = await apiClient.get('/admin/audit-logs', {
        params: { timeRange: filterOptions.timeRange },
      });
      return response.data;
    }
  );
  
  // Fetch backups
  const { data: backups, isLoading: backupsLoading } = useQuery(
    'backups',
    async () => {
      const response = await apiClient.get('/admin/backups');
      return response.data;
    }
  );
  
  // Fetch integrations
  const { data: integrations, isLoading: integrationsLoading } = useQuery(
    'integrations',
    async () => {
      const response = await apiClient.get('/admin/integrations');
      return response.data;
    }
  );
  
  // Mutations
  const updateConfigMutation = useMutation(
    async (config: Configuration) => {
      const response = await apiClient.put(`/admin/configurations/${config.id}`, config);
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('configurations');
        toast.success('Configuration updated successfully');
        setConfigDialogOpen(false);
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to update configuration');
      },
    }
  );
  
  const saveUserMutation = useMutation(
    async (user: User) => {
      if (user.id) {
        const response = await apiClient.put(`/admin/users/${user.id}`, user);
        return response.data;
      } else {
        const response = await apiClient.post('/admin/users', user);
        return response.data;
      }
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        toast.success('User saved successfully');
        setUserDialogOpen(false);
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to save user');
      },
    }
  );
  
  const deleteUserMutation = useMutation(
    async (userId: string) => {
      await apiClient.delete(`/admin/users/${userId}`);
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        toast.success('User deleted successfully');
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to delete user');
      },
    }
  );
  
  const restartServiceMutation = useMutation(
    async (serviceName: string) => {
      await apiClient.post(`/admin/services/${serviceName}/restart`);
    },
    {
      onSuccess: (_, serviceName) => {
        queryClient.invalidateQueries('systemStatus');
        toast.success(`Service ${serviceName} restarted successfully`);
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to restart service');
      },
    }
  );
  
  const createBackupMutation = useMutation(
    async (backup: BackupJob) => {
      const response = await apiClient.post('/admin/backups', backup);
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('backups');
        toast.success('Backup created successfully');
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to create backup');
      },
    }
  );
  
  const testIntegrationMutation = useMutation(
    async (integrationId: string) => {
      const response = await apiClient.post(`/admin/integrations/${integrationId}/test`);
      return response.data;
    },
    {
      onSuccess: (data) => {
        if (data.success) {
          toast.success('Integration test successful');
        } else {
          toast.error(`Integration test failed: ${data.error}`);
        }
      },
      onError: (error: any) => {
        toast.error(error.response?.data?.message || 'Failed to test integration');
      },
    }
  );
  
  // Handlers
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };
  
  const handleConfigEdit = (config: Configuration) => {
    setSelectedConfig(config);
    setConfigDialogOpen(true);
  };
  
  const handleConfigSave = (values: any) => {
    if (selectedConfig) {
      updateConfigMutation.mutate({ ...selectedConfig, ...values });
    }
  };
  
  const handleUserEdit = (user: User) => {
    setSelectedUser(user);
    setUserDialogOpen(true);
  };
  
  const handleUserDelete = (userId: string) => {
    setConfirmDialog({
      open: true,
      title: 'Delete User',
      message: 'Are you sure you want to delete this user? This action cannot be undone.',
      onConfirm: () => {
        deleteUserMutation.mutate(userId);
        setConfirmDialog({ ...confirmDialog, open: false });
      },
    });
  };
  
  const handleServiceRestart = (serviceName: string) => {
    setConfirmDialog({
      open: true,
      title: 'Restart Service',
      message: `Are you sure you want to restart ${serviceName}? This may cause temporary disruption.`,
      onConfirm: () => {
        restartServiceMutation.mutate(serviceName);
        setConfirmDialog({ ...confirmDialog, open: false });
      },
    });
  };
  
  const handleBackupNow = () => {
    const backup: BackupJob = {
      id: '',
      name: `Manual Backup ${new Date().toISOString()}`,
      schedule: 'manual',
      lastRun: '',
      nextRun: '',
      status: 'active',
      size: 0,
      duration: 0,
      retention: 30,
    };
    createBackupMutation.mutate(backup);
  };
  
  const handleIntegrationTest = (integrationId: string) => {
    testIntegrationMutation.mutate(integrationId);
  };
  
  // File upload for restore
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const formData = new FormData();
      formData.append('backup', file);
      
      apiClient.post('/admin/restore', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(() => {
        toast.success('Restore initiated successfully');
      })
      .catch((error) => {
        toast.error(error.response?.data?.message || 'Failed to restore');
      });
    }
  }, []);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/x-tar': ['.tar', '.tar.gz'],
      'application/zip': ['.zip'],
    },
    maxFiles: 1,
  });
  
  // Dashboard Tab
  const DashboardTab = () => (
    <Box>
      <Grid container spacing={3}>
        {/* System Overview */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              System Overview
            </Typography>
            {statusLoading ? (
              <Skeleton variant="rectangular" height={200} />
            ) : (
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={2}>
                        <CheckCircle
                          color={systemStatus?.overall === 'healthy' ? 'success' : 'error'}
                          sx={{ mr: 1 }}
                        />
                        <Typography variant="h6">
                          System Status
                        </Typography>
                      </Box>
                      <Typography variant="h4" color={
                        systemStatus?.overall === 'healthy' ? 'success.main' :
                        systemStatus?.overall === 'degraded' ? 'warning.main' : 'error.main'
                      }>
                        {systemStatus?.overall?.toUpperCase()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={2}>
                        <Storage sx={{ mr: 1 }} />
                        <Typography variant="h6">
                          Active Scrapers
                        </Typography>
                      </Box>
                      <Typography variant="h4">
                        {systemStatus?.metrics?.activeScrapers || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        of {systemStatus?.metrics?.totalScrapers || 0} total
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={2}>
                        <TrendingUp sx={{ mr: 1 }} />
                        <Typography variant="h6">
                          Success Rate
                        </Typography>
                      </Box>
                      <Typography variant="h4">
                        {systemStatus?.metrics?.successRate?.toFixed(1) || 0}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={systemStatus?.metrics?.successRate || 0}
                        sx={{ mt: 1 }}
                      />
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={2}>
                        <Speed sx={{ mr: 1 }} />
                        <Typography variant="h6">
                          Avg Response Time
                        </Typography>
                      </Box>
                      <Typography variant="h4">
                        {systemStatus?.metrics?.averageResponseTime?.toFixed(0) || 0}ms
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Last 24 hours
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Paper>
        </Grid>
        
        {/* Services Status */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Services Status
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Service</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>CPU</TableCell>
                    <TableCell>Memory</TableCell>
                    <TableCell>Uptime</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {systemStatus?.services && Object.entries(systemStatus.services).map(([name, service]) => (
                    <TableRow key={name}>
                      <TableCell>{name}</TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          label={service.status}
                          color={
                            service.status === 'up' ? 'success' :
                            service.status === 'degraded' ? 'warning' : 'error'
                          }
                        />
                      </TableCell>
                      <TableCell>{service.cpu.toFixed(1)}%</TableCell>
                      <TableCell>{service.memory.toFixed(1)}%</TableCell>
                      <TableCell>{formatDistanceToNow(new Date(Date.now() - service.uptime * 1000))}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleServiceRestart(name)}
                          disabled={restartServiceMutation.isLoading}
                        >
                          <RestartAlt />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
        
        {/* Recent Activity */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <List sx={{ maxHeight: 320, overflow: 'auto' }}>
              {auditLogs?.slice(0, 10).map((log: AuditLog) => (
                <ListItem key={log.id}>
                  <ListItemIcon>
                    {log.result === 'success' ? <CheckCircle color="success" /> : <Error color="error" />}
                  </ListItemIcon>
                  <ListItemText
                    primary={log.action}
                    secondary={`${log.user} • ${formatDistanceToNow(parseISO(log.timestamp))} ago`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
  
  // Users Tab
  const UsersTab = () => {
    const columns: GridColDef[] = [
      { field: 'username', headerName: 'Username', width: 150 },
      { field: 'email', headerName: 'Email', width: 200 },
      {
        field: 'role',
        headerName: 'Role',
        width: 120,
        renderCell: (params) => (
          <Chip
            size="small"
            label={params.value}
            color={
              params.value === 'admin' ? 'error' :
              params.value === 'operator' ? 'warning' : 'default'
            }
          />
        ),
      },
      {
        field: 'status',
        headerName: 'Status',
        width: 120,
        renderCell: (params) => (
          <Chip
            size="small"
            label={params.value}
            color={params.value === 'active' ? 'success' : 'default'}
          />
        ),
      },
      {
        field: 'lastLogin',
        headerName: 'Last Login',
        width: 180,
        valueFormatter: (params) => params.value ? formatDistanceToNow(parseISO(params.value)) + ' ago' : 'Never',
      },
      {
        field: 'actions',
        headerName: 'Actions',
        width: 150,
        renderCell: (params) => (
          <>
            <IconButton size="small" onClick={() => handleUserEdit(params.row)}>
              <Edit />
            </IconButton>
            <IconButton size="small" onClick={() => handleUserDelete(params.row.id)}>
              <Delete />
            </IconButton>
          </>
        ),
      },
    ];
    
    return (
      <Box>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h5">User Management</Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => {
              setSelectedUser(null);
              setUserDialogOpen(true);
            }}
          >
            Add User
          </Button>
        </Box>
        
        <Paper sx={{ p: 2 }}>
          <Box display="flex" gap={2} mb={2}>
            <TextField
              placeholder="Search users..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <FormControl sx={{ minWidth: 120 }}>
              <Select
                value={filterOptions.role}
                onChange={(e) => setFilterOptions({ ...filterOptions, role: e.target.value })}
                size="small"
              >
                <MenuItem value="all">All Roles</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
                <MenuItem value="operator">Operator</MenuItem>
                <MenuItem value="viewer">Viewer</MenuItem>
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 120 }}>
              <Select
                value={filterOptions.status}
                onChange={(e) => setFilterOptions({ ...filterOptions, status: e.target.value })}
                size="small"
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="inactive">Inactive</MenuItem>
                <MenuItem value="suspended">Suspended</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <DataGrid
            rows={users || []}
            columns={columns}
            pageSize={10}
            rowsPerPageOptions={[10, 25, 50]}
            checkboxSelection
            disableSelectionOnClick
            autoHeight
            loading={usersLoading}
          />
        </Paper>
      </Box>
    );
  };
  
  // Configuration Tab
  const ConfigurationTab = () => {
    const groupedConfigs = useMemo(() => {
      if (!configurations) return {};
      return configurations.reduce((acc: any, config: Configuration) => {
        if (!acc[config.category]) acc[config.category] = [];
        acc[config.category].push(config);
        return acc;
      }, {});
    }, [configurations]);
    
    return (
      <Box>
        <Typography variant="h5" gutterBottom>
          System Configuration
        </Typography>
        
        {configsLoading ? (
          <Skeleton variant="rectangular" height={400} />
        ) : (
          Object.entries(groupedConfigs).map(([category, configs]: [string, any]) => (
            <Accordion key={category}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">{category}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Key</TableCell>
                        <TableCell>Value</TableCell>
                        <TableCell>Description</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {configs.map((config: Configuration) => (
                        <TableRow key={config.id}>
                          <TableCell>
                            <Typography variant="body2" fontFamily="monospace">
                              {config.key}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {config.type === 'boolean' ? (
                              <Chip
                                size="small"
                                label={config.value ? 'True' : 'False'}
                                color={config.value ? 'success' : 'default'}
                              />
                            ) : config.type === 'json' ? (
                              <Typography variant="body2" fontFamily="monospace">
                                {JSON.stringify(config.value)}
                              </Typography>
                            ) : (
                              <Typography variant="body2">
                                {config.value}
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {config.description}
                            </Typography>
                            {config.requiresRestart && (
                              <Chip
                                size="small"
                                label="Requires Restart"
                                color="warning"
                                sx={{ mt: 0.5 }}
                              />
                            )}
                          </TableCell>
                          <TableCell>
                            <IconButton
                              size="small"
                              onClick={() => handleConfigEdit(config)}
                              disabled={!config.editable}
                            >
                              <Edit />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </AccordionDetails>
            </Accordion>
          ))
        )}
      </Box>
    );
  };
  
  // Security Tab
  const SecurityTab = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        Security & Audit
      </Typography>
      
      <Grid container spacing={3}>
        {/* Security Overview */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Security Status
            </Typography>
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText
                  primary="SSL/TLS Enabled"
                  secondary="All connections are encrypted"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText
                  primary="2FA Enabled"
                  secondary="Two-factor authentication is active"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Warning color="warning" />
                </ListItemIcon>
                <ListItemText
                  primary="Password Policy"
                  secondary="3 users have weak passwords"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText
                  primary="API Rate Limiting"
                  secondary="Rate limiting is configured"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
        
        {/* Active Sessions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Sessions
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>User</TableCell>
                    <TableCell>IP Address</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>admin@example.com</TableCell>
                    <TableCell>192.168.1.100</TableCell>
                    <TableCell>2h 15m</TableCell>
                    <TableCell>
                      <IconButton size="small">
                        <Close />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
        
        {/* Audit Logs */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Audit Logs
              </Typography>
              <FormControl size="small">
                <Select
                  value={filterOptions.timeRange}
                  onChange={(e) => setFilterOptions({ ...filterOptions, timeRange: e.target.value })}
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                </Select>
              </FormControl>
            </Box>
            
            <TableContainer sx={{ maxHeight: 400 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>User</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell>Resource</TableCell>
                    <TableCell>IP Address</TableCell>
                    <TableCell>Result</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {auditLogs?.map((log: AuditLog) => (
                    <TableRow key={log.id}>
                      <TableCell>
                        {format(parseISO(log.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                      </TableCell>
                      <TableCell>{log.user}</TableCell>
                      <TableCell>{log.action}</TableCell>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {log.resource}
                        </Typography>
                      </TableCell>
                      <TableCell>{log.ip}</TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          label={log.result}
                          color={log.result === 'success' ? 'success' : 'error'}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
  
  // Backup & Restore Tab
  const BackupRestoreTab = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        Backup & Restore
      </Typography>
      
      <Grid container spacing={3}>
        {/* Backup Controls */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <Typography variant="h6">
                Backup Management
              </Typography>
              <Button
                variant="contained"
                startIcon={<CloudUpload />}
                onClick={handleBackupNow}
                disabled={createBackupMutation.isLoading}
              >
                Backup Now
              </Button>
            </Box>
            
            <List>
              {backups?.map((backup: BackupJob) => (
                <ListItem key={backup.id}>
                  <ListItemIcon>
                    <CloudDownload />
                  </ListItemIcon>
                  <ListItemText
                    primary={backup.name}
                    secondary={
                      <>
                        <Typography variant="body2" component="span">
                          {backup.schedule} • Last run: {formatDistanceToNow(parseISO(backup.lastRun))} ago
                        </Typography>
                        <br />
                        <Typography variant="body2" component="span">
                          Size: {(backup.size / 1024 / 1024).toFixed(2)} MB • Duration: {backup.duration}s
                        </Typography>
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end">
                      <Download />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
        
        {/* Restore */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Restore from Backup
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'divider',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s',
                '&:hover': {
                  borderColor: 'primary.main',
                  backgroundColor: alpha(theme.palette.primary.main, 0.04),
                },
              }}
            >
              <input {...getInputProps()} />
              <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="body1" gutterBottom>
                {isDragActive
                  ? 'Drop the backup file here...'
                  : 'Drag & drop a backup file here, or click to select'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Supported formats: .tar, .tar.gz, .zip
              </Typography>
            </Box>
            
            <Alert severity="warning" sx={{ mt: 2 }}>
              <AlertTitle>Warning</AlertTitle>
              Restoring from a backup will overwrite existing data. Make sure to backup current data before proceeding.
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
  
  // Integrations Tab
  const IntegrationsTab = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        Integrations
      </Typography>
      
      <Grid container spacing={3}>
        {integrations?.map((integration: Integration) => (
          <Grid item xs={12} md={6} lg={4} key={integration.id}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Avatar sx={{ bgcolor: integration.status === 'connected' ? 'success.main' : 'error.main' }}>
                    <Api />
                  </Avatar>
                  <Box ml={2}>
                    <Typography variant="h6">
                      {integration.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {integration.type}
                    </Typography>
                  </Box>
                </Box>
                
                <Chip
                  size="small"
                  label={integration.status}
                  color={integration.status === 'connected' ? 'success' : 'error'}
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="body2" gutterBottom>
                  Last sync: {formatDistanceToNow(parseISO(integration.lastSync))} ago
                </Typography>
                
                <Box display="flex" justifyContent="space-between" mt={2}>
                  <Typography variant="body2">
                    Synced: {integration.stats.totalSynced}
                  </Typography>
                  <Typography variant="body2">
                    Failed: {integration.stats.failedSyncs}
                  </Typography>
                  <Typography variant="body2">
                    Latency: {integration.stats.averageLatency}ms
                  </Typography>
                </Box>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => handleIntegrationTest(integration.id)}>
                  Test Connection
                </Button>
                <Button size="small">Configure</Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
  
  // Tab panels
  const tabPanels = [
    <DashboardTab key="dashboard" />,
    <UsersTab key="users" />,
    <ConfigurationTab key="configuration" />,
    <SecurityTab key="security" />,
    <BackupRestoreTab key="backup" />,
    <IntegrationsTab key="integrations" />,
  ];
  
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        {/* Sidebar */}
        <Drawer
          variant={isMobile ? 'temporary' : 'persistent'}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar>
            <Typography variant="h6" noWrap>
              Admin Panel
            </Typography>
          </Toolbar>
          <Divider />
          <List>
            {[
              { text: 'Dashboard', icon: <Dashboard />, value: 0 },
              { text: 'Users', icon: <People />, value: 1 },
              { text: 'Configuration', icon: <Settings />, value: 2 },
              { text: 'Security', icon: <Security />, value: 3 },
              { text: 'Backup', icon: <CloudUpload />, value: 4 },
              { text: 'Integrations', icon: <IntegrationInstructions />, value: 5 },
            ].map((item) => (
              <ListItem
                button
                key={item.text}
                selected={currentTab === item.value}
                onClick={() => setCurrentTab(item.value)}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Drawer>
        
        {/* Main content */}
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
              <IconButton
                color="inherit"
                edge="start"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                sx={{ mr: 2 }}
              >
                <Menu />
              </IconButton>
              <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                MCP Stack Administration - 40by6
              </Typography>
              
              {/* Notifications */}
              <IconButton
                color="inherit"
                onClick={(e) => setNotificationMenuAnchor(e.currentTarget)}
              >
                <Badge badgeContent={notifications.length} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
              <Menu
                anchorEl={notificationMenuAnchor}
                open={Boolean(notificationMenuAnchor)}
                onClose={() => setNotificationMenuAnchor(null)}
              >
                {notifications.length === 0 ? (
                  <MenuItem>No new notifications</MenuItem>
                ) : (
                  notifications.map((notif, index) => (
                    <MenuItem key={index} onClick={() => {
                      // Handle notification click
                      setNotificationMenuAnchor(null);
                    }}>
                      <ListItemText
                        primary={notif.title}
                        secondary={notif.message}
                      />
                    </MenuItem>
                  ))
                )}
              </Menu>
              
              {/* User menu */}
              <IconButton color="inherit">
                <AccountCircle />
              </IconButton>
            </Toolbar>
          </AppBar>
          
          <Toolbar /> {/* Spacer for fixed AppBar */}
          
          <Container maxWidth={false}>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
              >
                {tabPanels[currentTab]}
              </motion.div>
            </AnimatePresence>
          </Container>
        </Box>
        
        {/* Dialogs */}
        <Dialog
          open={configDialogOpen}
          onClose={() => setConfigDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Edit Configuration</DialogTitle>
          <DialogContent>
            {selectedConfig && (
              <Box sx={{ pt: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {selectedConfig.description}
                </Typography>
                {selectedConfig.type === 'boolean' ? (
                  <FormControlLabel
                    control={
                      <Switch
                        defaultChecked={selectedConfig.value}
                        onChange={(e) => {
                          selectedConfig.value = e.target.checked;
                        }}
                      />
                    }
                    label={selectedConfig.key}
                  />
                ) : selectedConfig.type === 'number' ? (
                  <TextField
                    fullWidth
                    type="number"
                    label={selectedConfig.key}
                    defaultValue={selectedConfig.value}
                    onChange={(e) => {
                      selectedConfig.value = parseFloat(e.target.value);
                    }}
                  />
                ) : selectedConfig.type === 'list' ? (
                  <Autocomplete
                    multiple
                    options={[]}
                    freeSolo
                    defaultValue={selectedConfig.value}
                    onChange={(_, newValue) => {
                      selectedConfig.value = newValue;
                    }}
                    renderInput={(params) => (
                      <TextField {...params} label={selectedConfig.key} />
                    )}
                  />
                ) : (
                  <TextField
                    fullWidth
                    label={selectedConfig.key}
                    defaultValue={selectedConfig.value}
                    multiline={selectedConfig.type === 'json'}
                    rows={selectedConfig.type === 'json' ? 4 : 1}
                    onChange={(e) => {
                      selectedConfig.value = selectedConfig.type === 'json'
                        ? JSON.parse(e.target.value)
                        : e.target.value;
                    }}
                  />
                )}
                {selectedConfig.requiresRestart && (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    This configuration change requires a service restart to take effect.
                  </Alert>
                )}
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={() => handleConfigSave(selectedConfig)}
              variant="contained"
              disabled={updateConfigMutation.isLoading}
            >
              Save
            </Button>
          </DialogActions>
        </Dialog>
        
        <Dialog
          open={userDialogOpen}
          onClose={() => setUserDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>
            {selectedUser ? 'Edit User' : 'Add User'}
          </DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                fullWidth
                label="Username"
                defaultValue={selectedUser?.username}
                onChange={(e) => {
                  if (selectedUser) selectedUser.username = e.target.value;
                }}
              />
              <TextField
                fullWidth
                label="Email"
                type="email"
                defaultValue={selectedUser?.email}
                onChange={(e) => {
                  if (selectedUser) selectedUser.email = e.target.value;
                }}
              />
              <FormControl fullWidth>
                <InputLabel>Role</InputLabel>
                <Select
                  value={selectedUser?.role || 'viewer'}
                  onChange={(e) => {
                    if (selectedUser) selectedUser.role = e.target.value as any;
                  }}
                >
                  <MenuItem value="admin">Admin</MenuItem>
                  <MenuItem value="operator">Operator</MenuItem>
                  <MenuItem value="viewer">Viewer</MenuItem>
                </Select>
              </FormControl>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={selectedUser?.status || 'active'}
                  onChange={(e) => {
                    if (selectedUser) selectedUser.status = e.target.value as any;
                  }}
                >
                  <MenuItem value="active">Active</MenuItem>
                  <MenuItem value="inactive">Inactive</MenuItem>
                  <MenuItem value="suspended">Suspended</MenuItem>
                </Select>
              </FormControl>
              {!selectedUser && (
                <TextField
                  fullWidth
                  label="Password"
                  type="password"
                  helperText="Minimum 8 characters with mixed case and numbers"
                />
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setUserDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={() => {
                if (selectedUser) {
                  saveUserMutation.mutate(selectedUser);
                }
              }}
              variant="contained"
              disabled={saveUserMutation.isLoading}
            >
              Save
            </Button>
          </DialogActions>
        </Dialog>
        
        <Dialog
          open={confirmDialog.open}
          onClose={() => setConfirmDialog({ ...confirmDialog, open: false })}
        >
          <DialogTitle>{confirmDialog.title}</DialogTitle>
          <DialogContent>
            <DialogContentText>{confirmDialog.message}</DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setConfirmDialog({ ...confirmDialog, open: false })}>
              Cancel
            </Button>
            <Button onClick={confirmDialog.onConfirm} variant="contained" color="error">
              Confirm
            </Button>
          </DialogActions>
        </Dialog>
        
        <ToastContainer
          position="bottom-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="dark"
        />
      </Box>
    </ThemeProvider>
  );
};

export default AdminControlPanel;