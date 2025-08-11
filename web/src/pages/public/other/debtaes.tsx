import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import DebateCard from '../../../components/DebateCard';
import api from '../../../api/axios';
import DebateCardSkeleton from '../../../components/DebateCardSkeleton';

function DebateSearch() {
  const location = useLocation();
  const fullPath = location.pathname;

  const segments = location.pathname.split('/').filter(Boolean);
  const lastSegment = segments[segments.length - 1];

  const isValid = /^[a-z\-]+-\d+$/.test(lastSegment);

  const parts = lastSegment.split('-');
  const speech_number = parts[parts.length - 1];

  const [data, setData] = useState([]);
  const [politician, setPolitician] = useState('');

  const loadHouseMentions = async () => {
    // console.log(speech_number);
    try {
      const response = await api.get('/web/get-house-mention', {
        params: {
          params: fullPath,
          politician: isValid ? lastSegment : '',
        },
      });
      setData(response.data.data);
      setPolitician(response.data.politician);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    }
  };

  useEffect(() => {
    loadHouseMentions();
  }, []);

  return (
    <div className="padding-x">
      <div>
        <h1 className="text-[20px] md:text-[24px] font-[SFProTextBold] mt-12 md:mt-20 mb-6 md:mb-8">
          {politician} House Mentions
        </h1>

        <div className="space-y-4 md:space-y-6">
          {data.length
            ? data.map((item, index) => (
                <DebateCard
                  key={index}
                  debate={item}
                  index={index}
                  politician={politician}
                  speech_number={speech_number}
                  matchIndex={speech_number}
                />
              ))
            : Array.from({ length: 3 }).map((_, i) => (
                <DebateCardSkeleton key={i} />
              ))}
        </div>
      </div>
    </div>
  );
}

export default DebateSearch;
