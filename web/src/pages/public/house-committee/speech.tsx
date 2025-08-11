import { useEffect, useState } from 'react';
import DebateCard from '../../../components/DebateCard';
import api from '../../../api/axios';

const Speech = () => {
  const [houseMentions, setHouseMentions] = useState<any[]>([]);

  const LoadTopics = async () => {
    try {
      const response_mention = await api.get(
        '/web/committee/committee-mentions/',
      );
      setHouseMentions(response_mention.data.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    }
  };
  useEffect(() => {
    LoadTopics();
  }, []);
  return (
    <div className="pt-[1rem]">
      <div className="grid md:grid-cols-2 grid-cols-1 gap-4 padding-x bg-white pb-4">
        <h1 className="text-[32px] md:text-[40px] text-[#222222] font-['SFProDisplay']">
          Speech <span className="text-[#AFAFAF]">Committees</span>
        </h1>
      </div>
      <div className="padding-x mt-8 space-y-6">
        {houseMentions.map((houseMentions, index) => (
          <DebateCard key={index} debate={houseMentions} index={index} />
        ))}
      </div>
    </div>
  );
};

export default Speech;
