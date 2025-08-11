import { useEffect, useState } from 'react';
import DebateCard from '../../../components/DebateCard';
import Select from '../../../components/common/Select';
import api from '../../../api/axios';
import DebateCardSkeleton from '../../../components/DebateCardSkeleton';

const HouseCommitte = () => {
  const [days, setDays] = useState([]);
  const [years, setYears] = useState([]);
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState('');
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedDay, setSelectedDay] = useState('');
  const [houseMentions, setHouseMentions] = useState<any[]>([]);

  const loadTopicsAndMentions = async () => {
    try {
      setDays([]);
      setYears([]);
      setHouseMentions([]);
      setSelectedTopic('');
      setSelectedYear('');
      setSelectedDay('');

      const topicsResponse = await api.get('/web/committee/committee-topics');
      setTopics(topicsResponse.data.data);

      // Initially load mentions without filters (or first topic if exists)
      const mentionsResponse = await api.get(
        '/web/committee/committee-mentions/',
      );
      setHouseMentions(mentionsResponse.data.data);
    } catch (error) {
      // Handle error if needed
    }
  };

  // When topic changes, load years for that topic and reset dependent states
  const loadYearsForTopic = async (topicId: string) => {
    try {
      setSelectedTopic(topicId);
      setYears([]);
      setDays([]);
      setHouseMentions([]);
      setSelectedYear('');
      setSelectedDay('');

      const response = await api.get('/web/committee/committee-get-year', {
        params: { id: topicId },
      });
      setYears(response.data.data);
    } catch (error) {
      // Handle error if needed
    }
  };

  // When year changes, load days for that year and topic
  const loadDaysForYear = async (year: string) => {
    try {
      setSelectedYear(year);
      setDays([]);
      setHouseMentions([]);
      setSelectedDay('');

      // Assuming API expects topicId + year or just year
      const response = await api.get(
        '/web/committee/committee-get-year-data/' + year,
        {
          params: { year, topicId: selectedTopic },
        },
      );
      setDays(response.data.data);
    } catch (error) {
      // Handle error if needed
    }
  };

  // When day changes, load mentions for that day
  const loadMentionsForDay = async (day: string) => {
    try {
      setSelectedDay(day);
      setHouseMentions([]);

      const response = await api.get(
        '/web/committee/committee-mentions/' + day,
      );
      setHouseMentions(response.data.data);
    } catch (error) {
      // Handle error if needed
    }
  };

  useEffect(() => {
    loadTopicsAndMentions();
  }, []);

  return (
    <div className="pt-[1rem]">
      <div className="grid md:grid-cols-2 grid-cols-1 gap-4 padding-x bg-white pb-4">
        <h1 className="text-[32px] md:text-[40px] text-[#222222] font-['SFProDisplay']">
          House <span className="text-[#AFAFAF]">Committees</span>
        </h1>
      </div>

      <div className="sticky top-[80px] z-10 bg-white">
        <div className="bg-[#F4F4F4] min-h-[92px] w-full padding-x flex flex-col md:flex-row items-center justify-between py-4 md:py-0 gap-4 md:gap-0">
          <Select
            options={topics}
            placeholder="Select Current Committees"
            width="w-full max-w-[628px]"
            value={selectedTopic}
            onChange={loadYearsForTopic}
          />

          <div className="flex items-center gap-3 flex-wrap justify-center md:justify-end w-full md:w-auto">
            <Select
              options={years}
              placeholder="Select Year"
              width="w-full md:w-[120px]"
              value={selectedYear}
              onChange={loadDaysForYear}
            />
            <Select
              options={days}
              placeholder="Pick a Date"
              width="w-full md:w-[150px]"
              value={selectedDay}
              onChange={loadMentionsForDay}
            />
          </div>
        </div>
      </div>

      <div className="padding-x mt-8 space-y-6">
        {houseMentions.length
          ? houseMentions.map((mention, index) => (
              <DebateCard key={index} debate={mention} index={index} />
            ))
          : Array.from({ length: 3 }).map((_, i) => (
              <DebateCardSkeleton key={i} />
            ))}
      </div>
    </div>
  );
};

export default HouseCommitte;
