"""
MCP Stack Mobile Application - 40by6
Main React Native application with Expo
"""

import React, { useEffect, useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Dimensions,
  Platform,
  StatusBar,
  ActivityIndicator,
  Alert,
  Animated,
  Vibration,
  Linking,
} from 'react-native';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { SafeAreaProvider, SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Provider as PaperProvider, MD3DarkTheme, MD3LightTheme, adaptNavigationTheme } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';
import * as LocalAuthentication from 'expo-local-authentication';
import * as Notifications from 'expo-notifications';
import * as Updates from 'expo-updates';
import * as Device from 'expo-device';
import * as Haptics from 'expo-haptics';
import * as SplashScreen from 'expo-splash-screen';
import * as Font from 'expo-font';
import { Camera } from 'expo-camera';
import { BarCodeScanner } from 'expo-barcode-scanner';
import { Audio } from 'expo-av';
import NetInfo from '@react-native-community/netinfo';
import { GestureHandlerRootView, PanGestureHandler, State } from 'react-native-gesture-handler';
import { LineChart, BarChart, PieChart, ProgressChart } from 'react-native-chart-kit';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';
import { BlurView } from 'expo-blur';
import LottieView from 'lottie-react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@expo/vector-icons/Ionicons';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import { useColorScheme } from 'react-native';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from 'react-query';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import axios from 'axios';
import * as Sentry from 'sentry-expo';
import Constants from 'expo-constants';
import { enableScreens } from 'react-native-screens';

// Enable screens for better performance
enableScreens();

// Keep the splash screen visible while we fetch resources
SplashScreen.preventAutoHideAsync();

// Initialize Sentry
Sentry.init({
  dsn: Constants.expoConfig?.extra?.sentryDsn,
  enableInExpoDevelopment: true,
  debug: __DEV__,
});

// Configure notifications
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// Screen dimensions
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// API Configuration
const API_BASE_URL = Constants.expoConfig?.extra?.apiUrl || 'https://api.openpolicy.me';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-Platform': Platform.OS,
    'X-App-Version': Constants.expoConfig?.version,
  },
});

// Add auth interceptor
apiClient.interceptors.request.use(async (config) => {
  const token = await SecureStore.getItemAsync('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Types
interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  avatar?: string;
}

interface SystemStatus {
  overall: 'healthy' | 'degraded' | 'critical';
  services: {
    [key: string]: {
      status: 'up' | 'down';
      uptime: number;
      cpu: number;
      memory: number;
    };
  };
  metrics: {
    totalScrapers: number;
    activeScrapers: number;
    successRate: number;
    dataProcessed: number;
  };
}

interface Scraper {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  lastRun?: string;
  nextRun?: string;
  successRate: number;
  dataCollected: number;
}

// Zustand Store
interface AppState {
  user: User | null;
  theme: 'light' | 'dark' | 'auto';
  biometricEnabled: boolean;
  notificationsEnabled: boolean;
  offlineMode: boolean;
  setUser: (user: User | null) => void;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  setBiometric: (enabled: boolean) => void;
  setNotifications: (enabled: boolean) => void;
  setOfflineMode: (enabled: boolean) => void;
  hydrate: () => Promise<void>;
}

const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      user: null,
      theme: 'auto',
      biometricEnabled: false,
      notificationsEnabled: true,
      offlineMode: false,
      setUser: (user) => set({ user }),
      setTheme: (theme) => set({ theme }),
      setBiometric: (enabled) => set({ biometricEnabled: enabled }),
      setNotifications: (enabled) => set({ notificationsEnabled: enabled }),
      setOfflineMode: (enabled) => set({ offlineMode: enabled }),
      hydrate: async () => {
        // Hydrate from secure storage
        const token = await SecureStore.getItemAsync('auth_token');
        if (token) {
          try {
            const response = await apiClient.get('/auth/me');
            set({ user: response.data });
          } catch (error) {
            await SecureStore.deleteItemAsync('auth_token');
          }
        }
      },
    }),
    {
      name: 'mcp-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);

// Create navigation
const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();
const Drawer = createDrawerNavigator();

// Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});

// Dashboard Screen
const DashboardScreen: React.FC = () => {
  const insets = useSafeAreaInsets();
  const [refreshing, setRefreshing] = useState(false);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  
  // Fetch system status
  const { data: systemStatus, isLoading, refetch } = useQuery<SystemStatus>(
    'systemStatus',
    async () => {
      const response = await apiClient.get('/api/v1/system/status');
      return response.data;
    },
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  );
  
  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      useNativeDriver: true,
    }).start();
  }, []);
  
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refetch();
    setRefreshing(false);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, [refetch]);
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'up':
        return '#4CAF50';
      case 'degraded':
        return '#FF9800';
      case 'critical':
      case 'down':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };
  
  const chartConfig = {
    backgroundGradientFrom: '#1E2923',
    backgroundGradientFromOpacity: 0,
    backgroundGradientTo: '#08130D',
    backgroundGradientToOpacity: 0.5,
    color: (opacity = 1) => `rgba(26, 255, 146, ${opacity})`,
    strokeWidth: 2,
    barPercentage: 0.5,
    useShadowColorFromDataset: false,
  };
  
  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#90caf9" />
        <Text style={styles.loadingText}>Loading dashboard...</Text>
      </View>
    );
  }
  
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { paddingBottom: insets.bottom }]}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#90caf9"
          />
        }
      >
        <Animated.View style={{ opacity: fadeAnim }}>
          {/* System Status Card */}
          <LinearGradient
            colors={['#1e3c72', '#2a5298']}
            style={styles.card}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>System Status</Text>
              <View
                style={[
                  styles.statusIndicator,
                  { backgroundColor: getStatusColor(systemStatus?.overall || 'unknown') },
                ]}
              />
            </View>
            <Text style={styles.statusText}>
              {systemStatus?.overall?.toUpperCase() || 'UNKNOWN'}
            </Text>
            
            {/* Metrics */}
            <View style={styles.metricsGrid}>
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>
                  {systemStatus?.metrics?.activeScrapers || 0}
                </Text>
                <Text style={styles.metricLabel}>Active Scrapers</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>
                  {systemStatus?.metrics?.successRate?.toFixed(1) || 0}%
                </Text>
                <Text style={styles.metricLabel}>Success Rate</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>
                  {(systemStatus?.metrics?.dataProcessed / 1024 / 1024).toFixed(1) || 0} MB
                </Text>
                <Text style={styles.metricLabel}>Data Processed</Text>
              </View>
            </View>
          </LinearGradient>
          
          {/* Performance Chart */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Performance Trends</Text>
            <LineChart
              data={{
                labels: ['12h', '8h', '4h', '2h', '1h', 'Now'],
                datasets: [
                  {
                    data: [88, 92, 85, 91, 93, 95],
                    color: (opacity = 1) => `rgba(134, 65, 244, ${opacity})`,
                    strokeWidth: 2,
                  },
                ],
              }}
              width={SCREEN_WIDTH - 40}
              height={220}
              chartConfig={chartConfig}
              bezier
              style={styles.chart}
            />
          </View>
          
          {/* Services Grid */}
          <Text style={styles.sectionTitle}>Services</Text>
          <View style={styles.servicesGrid}>
            {systemStatus?.services &&
              Object.entries(systemStatus.services).map(([name, service]) => (
                <TouchableOpacity
                  key={name}
                  style={styles.serviceCard}
                  onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    Alert.alert(
                      name,
                      `Status: ${service.status}\nCPU: ${service.cpu.toFixed(1)}%\nMemory: ${service.memory.toFixed(1)}%`,
                      [{ text: 'OK' }]
                    );
                  }}
                >
                  <View
                    style={[
                      styles.serviceStatus,
                      { backgroundColor: getStatusColor(service.status) },
                    ]}
                  />
                  <Text style={styles.serviceName}>{name}</Text>
                  <Text style={styles.serviceMetric}>{service.cpu.toFixed(0)}%</Text>
                </TouchableOpacity>
              ))}
          </View>
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};

// Scrapers Screen
const ScrapersScreen: React.FC = () => {
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'active' | 'error'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  const { data: scrapers, isLoading } = useQuery<Scraper[]>(
    ['scrapers', selectedFilter],
    async () => {
      const response = await apiClient.get('/api/v1/scrapers', {
        params: { status: selectedFilter === 'all' ? undefined : selectedFilter },
      });
      return response.data;
    }
  );
  
  const toggleScraperMutation = useMutation(
    async (scraperId: string) => {
      await apiClient.post(`/api/v1/scrapers/${scraperId}/toggle`);
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('scrapers');
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      },
    }
  );
  
  const filteredScrapers = scrapers?.filter((scraper) =>
    scraper.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Scrapers</Text>
        <TouchableOpacity onPress={() => Alert.alert('Add Scraper', 'Feature coming soon!')}>
          <Ionicons name="add-circle" size={28} color="#90caf9" />
        </TouchableOpacity>
      </View>
      
      {/* Search Bar */}
      <View style={styles.searchBar}>
        <Ionicons name="search" size={20} color="#999" />
        <TextInput
          style={styles.searchInput}
          placeholder="Search scrapers..."
          placeholderTextColor="#999"
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>
      
      {/* Filter Tabs */}
      <View style={styles.filterTabs}>
        {(['all', 'active', 'error'] as const).map((filter) => (
          <TouchableOpacity
            key={filter}
            style={[
              styles.filterTab,
              selectedFilter === filter && styles.filterTabActive,
            ]}
            onPress={() => setSelectedFilter(filter)}
          >
            <Text
              style={[
                styles.filterTabText,
                selectedFilter === filter && styles.filterTabTextActive,
              ]}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      
      {/* Scrapers List */}
      <FlatList
        data={filteredScrapers}
        keyExtractor={(item) => item.id}
        refreshing={isLoading}
        onRefresh={() => queryClient.invalidateQueries('scrapers')}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.scraperItem}>
            <View style={styles.scraperHeader}>
              <View>
                <Text style={styles.scraperName}>{item.name}</Text>
                <Text style={styles.scraperType}>{item.type}</Text>
              </View>
              <Switch
                value={item.status === 'active'}
                onValueChange={() => toggleScraperMutation.mutate(item.id)}
                trackColor={{ false: '#767577', true: '#81b0ff' }}
                thumbColor={item.status === 'active' ? '#2196F3' : '#f4f3f4'}
              />
            </View>
            
            <View style={styles.scraperStats}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{item.successRate.toFixed(1)}%</Text>
                <Text style={styles.statLabel}>Success</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {(item.dataCollected / 1024).toFixed(1)}K
                </Text>
                <Text style={styles.statLabel}>Records</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {item.lastRun ? formatTime(item.lastRun) : 'Never'}
                </Text>
                <Text style={styles.statLabel}>Last Run</Text>
              </View>
            </View>
            
            <ProgressChart
              data={{
                labels: ['Success'],
                data: [item.successRate / 100],
              }}
              width={SCREEN_WIDTH - 60}
              height={100}
              strokeWidth={16}
              radius={32}
              chartConfig={{
                ...chartConfig,
                color: (opacity = 1) =>
                  item.successRate > 90
                    ? `rgba(76, 175, 80, ${opacity})`
                    : item.successRate > 70
                    ? `rgba(255, 152, 0, ${opacity})`
                    : `rgba(244, 67, 54, ${opacity})`,
              }}
              hideLegend={false}
              style={styles.progressChart}
            />
          </TouchableOpacity>
        )}
        ItemSeparatorComponent={() => <View style={styles.separator} />}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No scrapers found</Text>
        }
      />
    </SafeAreaView>
  );
};

// Analytics Screen
const AnalyticsScreen: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h');
  const [selectedMetric, setSelectedMetric] = useState<'runs' | 'data' | 'errors'>('runs');
  
  const { data: analytics } = useQuery(
    ['analytics', timeRange, selectedMetric],
    async () => {
      const response = await apiClient.get('/api/v1/analytics', {
        params: { timeRange, metric: selectedMetric },
      });
      return response.data;
    }
  );
  
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        <Text style={styles.headerTitle}>Analytics</Text>
        
        {/* Time Range Selector */}
        <View style={styles.timeRangeSelector}>
          {(['24h', '7d', '30d'] as const).map((range) => (
            <TouchableOpacity
              key={range}
              style={[
                styles.timeRangeButton,
                timeRange === range && styles.timeRangeButtonActive,
              ]}
              onPress={() => setTimeRange(range)}
            >
              <Text
                style={[
                  styles.timeRangeText,
                  timeRange === range && styles.timeRangeTextActive,
                ]}
              >
                {range === '24h' ? 'Last 24 Hours' : range === '7d' ? 'Last 7 Days' : 'Last 30 Days'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        
        {/* Summary Cards */}
        <View style={styles.summaryGrid}>
          <LinearGradient
            colors={['#667eea', '#764ba2']}
            style={styles.summaryCard}
          >
            <FontAwesome5 name="rocket" size={24} color="white" />
            <Text style={styles.summaryValue}>{analytics?.totalRuns || 0}</Text>
            <Text style={styles.summaryLabel}>Total Runs</Text>
          </LinearGradient>
          
          <LinearGradient
            colors={['#f093fb', '#f5576c']}
            style={styles.summaryCard}
          >
            <FontAwesome5 name="database" size={24} color="white" />
            <Text style={styles.summaryValue}>
              {((analytics?.dataCollected || 0) / 1024 / 1024).toFixed(1)} GB
            </Text>
            <Text style={styles.summaryLabel}>Data Collected</Text>
          </LinearGradient>
        </View>
        
        {/* Main Chart */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Performance Over Time</Text>
          <BarChart
            data={{
              labels: analytics?.chartLabels || [],
              datasets: [
                {
                  data: analytics?.chartData || [],
                },
              ],
            }}
            width={SCREEN_WIDTH - 40}
            height={250}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              style: {
                borderRadius: 16,
              },
            }}
            style={styles.chart}
            showValuesOnTopOfBars
          />
        </View>
        
        {/* Scraper Performance */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Top Performing Scrapers</Text>
          {analytics?.topScrapers?.map((scraper: any, index: number) => (
            <View key={scraper.id} style={styles.rankItem}>
              <Text style={styles.rankNumber}>#{index + 1}</Text>
              <View style={styles.rankInfo}>
                <Text style={styles.rankName}>{scraper.name}</Text>
                <View style={styles.rankStats}>
                  <Text style={styles.rankStat}>
                    {scraper.runs} runs • {scraper.successRate.toFixed(1)}% success
                  </Text>
                </View>
              </View>
              <Text style={styles.rankValue}>{scraper.dataCollected} records</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

// Settings Screen
const SettingsScreen: React.FC = ({ navigation }: any) => {
  const { user, theme, biometricEnabled, notificationsEnabled, setTheme, setBiometric, setNotifications, setUser } = useStore();
  const colorScheme = useColorScheme();
  const [biometricAvailable, setBiometricAvailable] = useState(false);
  
  useEffect(() => {
    checkBiometricAvailability();
  }, []);
  
  const checkBiometricAvailability = async () => {
    const compatible = await LocalAuthentication.hasHardwareAsync();
    const enrolled = await LocalAuthentication.isEnrolledAsync();
    setBiometricAvailable(compatible && enrolled);
  };
  
  const handleBiometricToggle = async (value: boolean) => {
    if (value && biometricAvailable) {
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate to enable biometric login',
        fallbackLabel: 'Use Passcode',
      });
      
      if (result.success) {
        setBiometric(true);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } else {
      setBiometric(false);
    }
  };
  
  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await SecureStore.deleteItemAsync('auth_token');
            setUser(null);
            navigation.replace('Auth');
          },
        },
      ],
      { cancelable: true }
    );
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        <Text style={styles.headerTitle}>Settings</Text>
        
        {/* User Profile */}
        {user && (
          <TouchableOpacity style={styles.profileSection}>
            <View style={styles.profileAvatar}>
              <Text style={styles.profileAvatarText}>
                {user.name.charAt(0).toUpperCase()}
              </Text>
            </View>
            <View style={styles.profileInfo}>
              <Text style={styles.profileName}>{user.name}</Text>
              <Text style={styles.profileEmail}>{user.email}</Text>
              <Text style={styles.profileRole}>{user.role}</Text>
            </View>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
        )}
        
        {/* Appearance */}
        <View style={styles.settingsSection}>
          <Text style={styles.sectionTitle}>Appearance</Text>
          <View style={styles.settingItem}>
            <Text style={styles.settingLabel}>Theme</Text>
            <View style={styles.themeSelector}>
              {(['light', 'dark', 'auto'] as const).map((t) => (
                <TouchableOpacity
                  key={t}
                  style={[
                    styles.themeOption,
                    theme === t && styles.themeOptionActive,
                  ]}
                  onPress={() => setTheme(t)}
                >
                  <Text
                    style={[
                      styles.themeOptionText,
                      theme === t && styles.themeOptionTextActive,
                    ]}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>
        
        {/* Security */}
        <View style={styles.settingsSection}>
          <Text style={styles.sectionTitle}>Security</Text>
          <View style={styles.settingItem}>
            <View>
              <Text style={styles.settingLabel}>Biometric Login</Text>
              {!biometricAvailable && (
                <Text style={styles.settingDescription}>
                  Not available on this device
                </Text>
              )}
            </View>
            <Switch
              value={biometricEnabled}
              onValueChange={handleBiometricToggle}
              disabled={!biometricAvailable}
              trackColor={{ false: '#767577', true: '#81b0ff' }}
              thumbColor={biometricEnabled ? '#2196F3' : '#f4f3f4'}
            />
          </View>
          
          <TouchableOpacity style={styles.settingItem}>
            <Text style={styles.settingLabel}>Change Password</Text>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.settingItem}>
            <Text style={styles.settingLabel}>Two-Factor Authentication</Text>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
        </View>
        
        {/* Notifications */}
        <View style={styles.settingsSection}>
          <Text style={styles.sectionTitle}>Notifications</Text>
          <View style={styles.settingItem}>
            <Text style={styles.settingLabel}>Push Notifications</Text>
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotifications}
              trackColor={{ false: '#767577', true: '#81b0ff' }}
              thumbColor={notificationsEnabled ? '#2196F3' : '#f4f3f4'}
            />
          </View>
          
          <TouchableOpacity style={styles.settingItem}>
            <Text style={styles.settingLabel}>Notification Settings</Text>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
        </View>
        
        {/* About */}
        <View style={styles.settingsSection}>
          <Text style={styles.sectionTitle}>About</Text>
          <TouchableOpacity style={styles.settingItem}>
            <Text style={styles.settingLabel}>Version</Text>
            <Text style={styles.settingValue}>{Constants.expoConfig?.version}</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.settingItem}
            onPress={() => Linking.openURL('https://openpolicy.me/privacy')}
          >
            <Text style={styles.settingLabel}>Privacy Policy</Text>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.settingItem}
            onPress={() => Linking.openURL('https://openpolicy.me/terms')}
          >
            <Text style={styles.settingLabel}>Terms of Service</Text>
            <Ionicons name="chevron-forward" size={24} color="#999" />
          </TouchableOpacity>
        </View>
        
        {/* Logout Button */}
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutButtonText}>Logout</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

// Auth Screen
const AuthScreen: React.FC = ({ navigation }: any) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const { setUser } = useStore();
  
  const handleAuth = async () => {
    if (!email || !password || (!isLogin && !name)) {
      Alert.alert('Error', 'Please fill all fields');
      return;
    }
    
    setLoading(true);
    
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const payload = isLogin ? { email, password } : { email, password, name };
      
      const response = await apiClient.post(endpoint, payload);
      const { token, user } = response.data;
      
      // Save token securely
      await SecureStore.setItemAsync('auth_token', token);
      
      // Update store
      setUser(user);
      
      // Navigate to main app
      navigation.replace('Main');
      
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error: any) {
      Alert.alert(
        'Authentication Failed',
        error.response?.data?.message || 'An error occurred'
      );
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <SafeAreaView style={styles.authContainer}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.authContent}
          keyboardShouldPersistTaps="handled"
        >
          <LottieView
            source={require('./assets/animations/logo.json')}
            autoPlay
            loop
            style={styles.logo}
          />
          
          <Text style={styles.authTitle}>
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </Text>
          <Text style={styles.authSubtitle}>
            {isLogin
              ? 'Sign in to access your MCP Stack'
              : 'Get started with MCP Stack'}
          </Text>
          
          {!isLogin && (
            <TextInput
              style={styles.input}
              placeholder="Full Name"
              placeholderTextColor="#999"
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
              textContentType="name"
            />
          )}
          
          <TextInput
            style={styles.input}
            placeholder="Email"
            placeholderTextColor="#999"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            textContentType="emailAddress"
          />
          
          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#999"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            textContentType={isLogin ? 'password' : 'newPassword'}
          />
          
          <TouchableOpacity
            style={[styles.authButton, loading && styles.authButtonDisabled]}
            onPress={handleAuth}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={styles.authButtonText}>
                {isLogin ? 'Sign In' : 'Sign Up'}
              </Text>
            )}
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.authSwitch}
            onPress={() => setIsLogin(!isLogin)}
          >
            <Text style={styles.authSwitchText}>
              {isLogin
                ? "Don't have an account? Sign Up"
                : 'Already have an account? Sign In'}
            </Text>
          </TouchableOpacity>
          
          {isLogin && (
            <TouchableOpacity style={styles.forgotPassword}>
              <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
            </TouchableOpacity>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

// Tab Navigator
const TabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string = 'home';
          
          if (route.name === 'Dashboard') {
            iconName = 'dashboard';
            return <MaterialIcons name={iconName} size={size} color={color} />;
          } else if (route.name === 'Scrapers') {
            iconName = 'web';
            return <MaterialIcons name={iconName} size={size} color={color} />;
          } else if (route.name === 'Analytics') {
            iconName = 'analytics';
            return <MaterialIcons name={iconName} size={size} color={color} />;
          } else if (route.name === 'Settings') {
            iconName = 'settings';
            return <Ionicons name={iconName} size={size} color={color} />;
          }
        },
        tabBarActiveTintColor: '#90caf9',
        tabBarInactiveTintColor: '#999',
        tabBarStyle: {
          backgroundColor: '#132f4c',
          borderTopColor: '#1e4976',
        },
        headerShown: false,
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Scrapers" component={ScrapersScreen} />
      <Tab.Screen name="Analytics" component={AnalyticsScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
};

// Main App Component
const App: React.FC = () => {
  const [isReady, setIsReady] = useState(false);
  const { user, theme, hydrate } = useStore();
  const colorScheme = useColorScheme();
  
  useEffect(() => {
    async function prepare() {
      try {
        // Load fonts
        await Font.loadAsync({
          'Inter-Regular': require('./assets/fonts/Inter-Regular.ttf'),
          'Inter-Medium': require('./assets/fonts/Inter-Medium.ttf'),
          'Inter-Bold': require('./assets/fonts/Inter-Bold.ttf'),
        });
        
        // Hydrate store
        await hydrate();
        
        // Check for updates
        if (!__DEV__) {
          const update = await Updates.checkForUpdateAsync();
          if (update.isAvailable) {
            await Updates.fetchUpdateAsync();
            await Updates.reloadAsync();
          }
        }
        
        // Request permissions
        const { status } = await Notifications.requestPermissionsAsync();
        if (status === 'granted') {
          const token = await Notifications.getExpoPushTokenAsync();
          console.log('Push token:', token);
        }
      } catch (e) {
        console.warn(e);
      } finally {
        setIsReady(true);
        await SplashScreen.hideAsync();
      }
    }
    
    prepare();
  }, []);
  
  // Network monitoring
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      if (!state.isConnected) {
        Alert.alert(
          'No Internet Connection',
          'Some features may be limited while offline.',
          [{ text: 'OK' }]
        );
      }
    });
    
    return () => unsubscribe();
  }, []);
  
  if (!isReady) {
    return null;
  }
  
  // Determine theme
  const isDarkMode =
    theme === 'dark' || (theme === 'auto' && colorScheme === 'dark');
  
  const customDarkTheme = {
    ...DarkTheme,
    colors: {
      ...DarkTheme.colors,
      primary: '#90caf9',
      background: '#0a1929',
      card: '#132f4c',
      text: '#ffffff',
      border: '#1e4976',
    },
  };
  
  const customLightTheme = {
    ...DefaultTheme,
    colors: {
      ...DefaultTheme.colors,
      primary: '#1976d2',
      background: '#ffffff',
      card: '#f5f5f5',
      text: '#000000',
      border: '#e0e0e0',
    },
  };
  
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <QueryClientProvider client={queryClient}>
        <PaperProvider theme={isDarkMode ? MD3DarkTheme : MD3LightTheme}>
          <SafeAreaProvider>
            <NavigationContainer
              theme={isDarkMode ? customDarkTheme : customLightTheme}
            >
              <StatusBar
                barStyle={isDarkMode ? 'light-content' : 'dark-content'}
                backgroundColor={isDarkMode ? '#0a1929' : '#ffffff'}
              />
              <Stack.Navigator screenOptions={{ headerShown: false }}>
                {user ? (
                  <Stack.Screen name="Main" component={TabNavigator} />
                ) : (
                  <Stack.Screen name="Auth" component={AuthScreen} />
                )}
              </Stack.Navigator>
            </NavigationContainer>
          </SafeAreaProvider>
        </PaperProvider>
      </QueryClientProvider>
    </GestureHandlerRootView>
  );
};

// Utility functions
const formatTime = (dateString: string) => {
  const date = new Date(dateString);
  const now = new Date();
  const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / 60000);
  
  if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
  if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
  return `${Math.floor(diffInMinutes / 1440)}d ago`;
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a1929',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollContent: {
    padding: 20,
  },
  loadingText: {
    color: '#90caf9',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 10,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 20,
  },
  card: {
    backgroundColor: '#132f4c',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 20,
  },
  metricsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metricItem: {
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#90caf9',
  },
  metricLabel: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 15,
    marginTop: 10,
  },
  servicesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  serviceCard: {
    backgroundColor: '#1e4976',
    borderRadius: 12,
    padding: 15,
    width: '48%',
    marginBottom: 15,
    alignItems: 'center',
  },
  serviceStatus: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginBottom: 8,
  },
  serviceName: {
    fontSize: 14,
    color: '#ffffff',
    marginBottom: 4,
  },
  serviceMetric: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#90caf9',
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1e4976',
    borderRadius: 12,
    paddingHorizontal: 15,
    marginHorizontal: 20,
    marginBottom: 15,
  },
  searchInput: {
    flex: 1,
    color: '#ffffff',
    fontSize: 16,
    paddingVertical: 12,
    paddingLeft: 10,
  },
  filterTabs: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  filterTab: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#1e4976',
    marginRight: 10,
  },
  filterTabActive: {
    backgroundColor: '#90caf9',
  },
  filterTabText: {
    color: '#999',
    fontSize: 14,
  },
  filterTabTextActive: {
    color: '#0a1929',
    fontWeight: 'bold',
  },
  scraperItem: {
    backgroundColor: '#132f4c',
    marginHorizontal: 20,
    padding: 20,
    borderRadius: 16,
  },
  scraperHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  scraperName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  scraperType: {
    fontSize: 14,
    color: '#999',
    marginTop: 2,
  },
  scraperStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#90caf9',
  },
  statLabel: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  progressChart: {
    alignSelf: 'center',
  },
  separator: {
    height: 15,
  },
  emptyText: {
    textAlign: 'center',
    color: '#999',
    fontSize: 16,
    marginTop: 50,
  },
  timeRangeSelector: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  timeRangeButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: '#1e4976',
    marginHorizontal: 5,
  },
  timeRangeButtonActive: {
    backgroundColor: '#90caf9',
  },
  timeRangeText: {
    color: '#999',
    fontSize: 14,
  },
  timeRangeTextActive: {
    color: '#0a1929',
    fontWeight: 'bold',
  },
  summaryGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    paddingHorizontal: 20,
  },
  summaryCard: {
    flex: 1,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginVertical: 10,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#ffffff',
    opacity: 0.8,
  },
  rankItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#1e4976',
  },
  rankNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#90caf9',
    width: 40,
  },
  rankInfo: {
    flex: 1,
  },
  rankName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  rankStats: {
    flexDirection: 'row',
  },
  rankStat: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  rankValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#90caf9',
  },
  profileSection: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#132f4c',
    padding: 20,
    borderRadius: 16,
    marginBottom: 20,
  },
  profileAvatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#90caf9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  profileAvatarText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#0a1929',
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  profileEmail: {
    fontSize: 14,
    color: '#999',
    marginTop: 2,
  },
  profileRole: {
    fontSize: 12,
    color: '#90caf9',
    marginTop: 4,
  },
  settingsSection: {
    marginBottom: 30,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#132f4c',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  settingLabel: {
    fontSize: 16,
    color: '#ffffff',
  },
  settingDescription: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  settingValue: {
    fontSize: 14,
    color: '#999',
  },
  themeSelector: {
    flexDirection: 'row',
  },
  themeOption: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#1e4976',
    marginLeft: 10,
  },
  themeOptionActive: {
    backgroundColor: '#90caf9',
  },
  themeOptionText: {
    color: '#999',
    fontSize: 14,
  },
  themeOptionTextActive: {
    color: '#0a1929',
    fontWeight: 'bold',
  },
  logoutButton: {
    backgroundColor: '#f44336',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 40,
  },
  logoutButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  authContainer: {
    flex: 1,
    backgroundColor: '#0a1929',
  },
  keyboardView: {
    flex: 1,
  },
  authContent: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  logo: {
    width: 150,
    height: 150,
    alignSelf: 'center',
    marginBottom: 40,
  },
  authTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 10,
  },
  authSubtitle: {
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
    marginBottom: 40,
  },
  input: {
    backgroundColor: '#132f4c',
    borderRadius: 12,
    paddingHorizontal: 20,
    paddingVertical: 15,
    fontSize: 16,
    color: '#ffffff',
    marginBottom: 15,
  },
  authButton: {
    backgroundColor: '#90caf9',
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 15,
  },
  authButtonDisabled: {
    opacity: 0.6,
  },
  authButtonText: {
    color: '#0a1929',
    fontSize: 18,
    fontWeight: 'bold',
  },
  authSwitch: {
    alignItems: 'center',
    marginTop: 10,
  },
  authSwitchText: {
    color: '#90caf9',
    fontSize: 14,
  },
  forgotPassword: {
    alignItems: 'center',
    marginTop: 20,
  },
  forgotPasswordText: {
    color: '#999',
    fontSize: 14,
  },
});

export default Sentry.Native.wrap(App);