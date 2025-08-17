import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import api from '../../../api/axios';
import DebateCardSkeleton from '../../../components/DebateCardSkeleton';

function BillSearch() {
  const location = useLocation();
  const fullPath = location.pathname.replace('/bills/', '');

  const navigate = useNavigate();

  const loadHouseMentions = async () => {
    try {
      const response = await api.get('/web/get-bill/', {
        params: {
          bill: fullPath,
        },
      });
      navigate(`/bills/${response.data.id}`, { replace: true });

      // setHouseMentions(response.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    }
  };

  useEffect(() => {
    loadHouseMentions();
  }, []);

  return <DebateCardSkeleton />;
}

export default BillSearch;
