import { createBrowserRouter, Navigate } from 'react-router-dom';
import App from './App';
import { AuthProvider } from './context/AuthContext';

// Public Pages
import Debates from './pages/public/debates';
import GovernmentBills from './pages/public/government-bills';
import HouseCommittee from './pages/public/house-committee';
import Mps from './pages/public/mps';
import Bills from './pages/public/bills';
import About from './pages/public/about';
import MPProfile from './pages/public/mps/[id]';
import FormerMps from './pages/public/mps/former-mp';
import DebateSearch from './pages/public/other/debates';
import BillSearch from './pages/public/other/bills';
import VotePage from './pages/public/vote';

// Admin Pages
import AdminDashboard from './pages/admin/dashboard';
import AdminScrapers from './pages/admin/scrapers';
import AdminPolicies from './pages/admin/policies';
import AdminSystem from './pages/admin/system';
import AdminLogin from './pages/admin/login';
import AdminApiDocs from './pages/admin/api-docs';

// Protected Route Component
import ProtectedRoute from './components/shared/ProtectedRoute';

export const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <AuthProvider>
        <App />
      </AuthProvider>
    ),
    children: [
      // Public Routes
      {
        path: '/',
        element: <Navigate to="/bills" replace />,
      },
      {
        path: '/bills',
        element: <GovernmentBills />,
      },
      {
        path: '/debates',
        element: <Debates />,
      },
      {
        path: '/committees',
        element: <HouseCommittee />,
      },
      {
        path: '/mps',
        element: <Mps />,
      },
      {
        path: '/former-mps',
        element: <FormerMps />,
      },
      {
        path: '/bills/:id',
        element: <Bills />,
      },
      {
        path: '/about',
        element: <About />,
      },
      {
        path: '/debates/*',
        element: <DebateSearch />,
      },
      {
        path: '/committees/*',
        element: <DebateSearch />,
      },
      {
        path: '/bills/*',
        element: <BillSearch />,
      },
      {
        path: '/mps/:id',
        element: <MPProfile />,
      },
      {
        path: '/votes/*',
        element: <VotePage />,
      },
      
      // Admin Routes (Protected)
      {
        path: '/admin/login',
        element: <AdminLogin />,
      },
      {
        path: '/admin',
        element: <ProtectedRoute />,
        children: [
          {
            path: '/admin',
            element: <Navigate to="/admin/dashboard" replace />,
          },
          {
            path: '/admin/dashboard',
            element: <AdminDashboard />,
          },
          {
            path: '/admin/scrapers',
            element: <AdminScrapers />,
          },
          {
            path: '/admin/policies',
            element: <AdminPolicies />,
          },
          {
            path: '/admin/system',
            element: <AdminSystem />,
          },
          {
            path: '/admin/api-docs',
            element: <AdminApiDocs />,
          },
        ],
      },
    ],
  },
]);

export default router;
