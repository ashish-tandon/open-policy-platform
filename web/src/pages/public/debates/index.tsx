import { useEffect, useState } from 'react';
import DebateCard from '../../../components/DebateCard';
import Select from '../../../components/common/Select';
import api from '../../../api/axios';
import DebateCardSkeleton from '../../../components/DebateCardSkeleton';
import EmptyState from '../../../components/common/EmptyState';
import { FiInbox } from 'react-icons/fi';

// const days = [
//   { value: 'feb14', label: 'February 14th' },
//   { value: 'feb13', label: 'February 13th' },
// ];

// const years = [{ value: '2025', label: '2025' }];

function Debates() {
  const [days, setDays] = useState([]);
  const [years, setYears] = useState([]);
  const [selectedDay, setSelectedDay] = useState('');
  const [selectedYear, setSelectedYear] = useState('');
  const [houseMentions, setHouseMentions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const loadYear = async () => {
    console.log(selectedYear);
    setDays([]);
    setLoading(true);
    try {
      const response = await api.get('/web/debate/debate-get-year');
      setYears(response.data.debate);

      const response_mention = await api.get(
        '/web/debate/debate-mentions/' + selectedDay,
      );
      setHouseMentions(response_mention.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    } finally {
      setLoading(false); // NEW
    }
  };

  const loadDays = async (value: any) => {
    try {
      const response = await api.get('/web/debate/debate-get-year-date', {
        params: {
          year: value,
        },
      });

      setDays(response.data.dates);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    }
  };

  const loadMentionData = async () => {
    setLoading(true);
    setHouseMentions([]);
    try {
      const response = await api.get(
        '/web/debate/debate-mentions/' + selectedDay,
      );
      setHouseMentions(response.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    } finally {
      setLoading(false); // NEW
    }
  };

  useEffect(() => {
    loadYear();
  }, []);

  return (
    <div className="pt-[1rem]">
      <div className="grid md:grid-cols-2 grid-cols-1 gap-4 padding-x bg-white pb-4">
        <h1 className="text-[32px] md:text-[40px] text-[#222222] font-['SFProDisplay']">
          The Debates of the{' '}
          <span className="text-[#AFAFAF]">House of Commons</span>
        </h1>
        <p className="text-[#222222] text-[14px] md:mt-0 mt-2">
          When Parliament is in session, every word spoken by a member is
          faithfully transcribed, and published in a document called a Hansard.
          We have the Hansards of the House of Commons dating back to 1994.
          Browse through them below, or search via the box above.
        </p>
      </div>
      <div className="sticky top-[80px] z-10 bg-white">
        <div className="bg-[#F4F4F4] min-h-[92px] w-full padding-x flex flex-col md:flex-row items-center justify-between py-4 md:py-0 gap-4 md:gap-0">
          {/* <Select
            options={topics}
            placeholder="Select A Topic"
            width="w-full max-w-[628px]"
          /> */}
          <span></span>
          <div className="flex items-center gap-3 flex-wrap justify-center md:justify-end w-full md:w-auto">
            <Select
              options={days}
              placeholder="Pick a Date"
              width="w-full md:w-[150px]"
              onChange={(value: string) => {
                setSelectedDay(value);
                loadMentionData();
              }}
            />
            <Select
              options={years}
              placeholder="Select Year"
              width="w-full md:w-[120px]"
              onChange={(value: string) => {
                setSelectedYear(value);
                loadDays(value);
              }}
            />
          </div>
        </div>
      </div>
      {houseMentions.length == 0 && loading == false && (
        <EmptyState
          icon={FiInbox}
          message="No items founds"
          description="Try adjusting your search or filters to find what you're looking for."
        />
      )}
      <div className="padding-x mt-8 space-y-6">
        {loading
          ? Array.from({ length: 3 }).map((_, i) => (
              <DebateCardSkeleton key={i} />
            ))
          : houseMentions.map((debate, index) => (
              <DebateCard key={index} debate={debate} index={index} />
            ))}
      </div>
    </div>
  );
}

export default Debates;
