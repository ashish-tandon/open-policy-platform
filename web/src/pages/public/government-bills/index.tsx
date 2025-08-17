import { IoSearchOutline } from 'react-icons/io5';
import GovernmentBillCard, {
  GovernmentBillCardProps,
} from '../../../components/GovernmentBillCard';
import Select from '../../../components/common/Select';
import { useEffect, useState } from 'react';
import api from '../../../api/axios';
import EmptyState from '../../../components/common/EmptyState';
import { FiInbox } from 'react-icons/fi';

const billOptions = [
  { label: 'All Bills', value: '' },
  { label: 'Government Bills', value: '1' },
  { label: 'Private Members Bills', value: '0' },
];

// Loading skeleton card to show while data is loading
function LoadingCard() {
  return (
    <div className="w-full h-[180px] rounded-2xl bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 animate-pulse shadow-sm p-4 flex flex-col justify-between">
      {/* Mimic card header */}
      <div className="h-6 bg-gray-300 rounded w-3/4 mb-4"></div>
      {/* Mimic card content lines */}
      <div className="space-y-2">
        <div className="h-4 bg-gray-300 rounded w-full"></div>
        <div className="h-4 bg-gray-300 rounded w-5/6"></div>
        <div className="h-4 bg-gray-300 rounded w-2/3"></div>
      </div>
      {/* Mimic card footer or button */}
      <div className="mt-6 h-8 bg-gray-300 rounded w-1/3"></div>
    </div>
  );
}

function GovernmentBills() {
  const [bills, setBills] = useState<GovernmentBillCardProps[]>([]);
  const [selected, setSelected] = useState('');
  const [selectedSession, setSelectedSession] = useState('');
  const [session, setSession] = useState([]);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [loading, setLoading] = useState(false);

  const loadBills = async () => {
    try {
      setLoading(true);
      const response = await api.get('/web/bills', {
        params: {
          search: debouncedSearch,
          bill_search: selected,
          session_search: selectedSession,
        },
      });

      setBills(response.data.bills);
      setSession(response.data.session);
    } catch (error) {
      // Optionally handle error here
      // showError(error.response?.data.message || error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(search);
    }, 300); // 300ms debounce delay

    return () => {
      clearTimeout(handler);
    };
  }, [search]);

  useEffect(() => {
    loadBills();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, selectedSession, debouncedSearch]);

  return (
    <div className="pt-[1rem]">
      <h1 className="text-[32px] md:text-[40px] text-[#222222] font-['SFProDisplay'] padding-x bg-white pb-[1rem]">
        Government <span className="text-[#AFAFAF]">Bills</span>
      </h1>
      <div className="sticky top-[80px] z-10 bg-white">
        <div className="bg-[#F4F4F4] min-h-[92px] w-full padding-x flex flex-col md:flex-row items-center justify-between py-4 md:py-0 gap-4 md:gap-0">
          <div className="max-w-[628px] w-full bg-white rounded-[24px] h-[44px] border border-[#EDEDED] flex items-center justify-between">
            <div className="flex items-center gap-3 w-full px-4">
              <img src="/assets/images/logo2.svg" alt="" className="w-6 h-6" />
              <input
                type="text"
                placeholder="Search Bills"
                className="placeholder:text-[#D3D3D3] w-full outline-none border-none"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <IoSearchOutline className="mr-4 cursor-pointer" size={20} />
          </div>
          <div className="flex items-center gap-3 flex-wrap justify-center md:justify-end w-full md:w-auto">
            <Select
              options={billOptions}
              placeholder="All Bills"
              width="w-full md:w-[150px]"
              onChange={(value: string) => setSelected(value)}
            />
            <Select
              options={session}
              placeholder="Sessions"
              width="w-full md:w-[120px]"
              onChange={(value: string) => setSelectedSession(value)}
            />
          </div>
        </div>
      </div>

      {/* Bills or Loading */}
      {bills.length == 0 && loading == false && (
        <EmptyState
          icon={FiInbox}
          message="No items founds"
          description="Try adjusting your search or filters to find what you're looking for."
        />
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-4 padding-x">
        {loading
          ? Array.from({ length: 6 }).map((_, index) => (
              <LoadingCard key={index} />
            ))
          : bills.map((bill) => <GovernmentBillCard key={bill.id} {...bill} />)}
      </div>
    </div>
  );
}

export default GovernmentBills;
