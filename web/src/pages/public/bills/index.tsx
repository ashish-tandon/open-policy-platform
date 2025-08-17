import { useParams } from 'react-router-dom';
import DebateCard from '../../../components/DebateCard';
import { useEffect, useState } from 'react';
import api from '../../../api/axios';
import DebateCardSkeleton from '../../../components/DebateCardSkeleton';

const Bills = () => {
  const { id } = useParams();

  interface BillData {
    bill_number: string;
    bill_type: string;
    name: string;
    short_name: string;
    sponsor: string;
    sponsor_party: string;
    status: string;
    summary: string;
    text_url: string;
    legisinfo_url: string;
    votes: {
      date: string;
      result: string;
      description: string;
    }[];
  }

  const [data, setData] = useState<BillData | null>(null);
  const [houseMentions, setHouseMentions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const loadBills = async () => {
    try {
      const response = await api.get('/web/bills/summary/' + id);
      setData(response.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHouseMentions = async () => {
    try {
      const response = await api.get('/web/bills/house-mention/' + id);
      // console.log(response.data);

      setHouseMentions(response.data);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    }
  };

  useEffect(() => {
    window.scrollTo(0, 0);
    loadBills();
    loadHouseMentions();
  }, [id]);

  return (
    <div className="pt-[2rem] padding-x">
      <div className="flex items-center gap-2 text-[24px] md:text-[32px] font-['SFProDisplay'] overflow-x-auto whitespace-nowrap pb-2">
        <h1 className="text-[#222222]">{data?.bill_type}</h1>
        <h1 className="text-[#AFAFAF]">Bills</h1>
        <span className="text-[#AFAFAF]">›</span>
        <h1 className="text-[#222222]">Bill {data?.bill_number}</h1>
      </div>

      <div className="mt-6 md:mt-8">
        <h2 className="text-[20px] md:text-[24px] font-['SFProDisplay'] text-[#222222] mb-2">
          {data?.name}
        </h2>
        <p className="text-[#515151] text-sm md:text-base max-w-[824px]">
          {data?.short_name}
        </p>
      </div>

      <div className="grid md:grid-cols-1 gap-6 md:gap-8 mt-6 md:mt-8 rounded-[24px] shadow-md p-4 md:p-6">
        {loading ? (
          <DebateCardSkeleton />
        ) : (
          <div className="space-y-4 md:space-y-6">
            <div className="grid grid-cols-[100px_1fr] md:grid-cols-[120px_1fr] gap-2 md:gap-4">
              <h3 className="text-[#000] font-['SFProTextMedium'] text-sm md:text-base">
                Sponsor
              </h3>
              <div className="flex flex-col md:flex-row md:items-center gap-1 md:gap-2">
                <p className="text-[#000] text-sm md:text-base">
                  {data?.sponsor}
                </p>
                <span
                  className="text-sm md:text-base"
                  style={{
                    color:
                      data?.sponsor_party == 'Bloc'
                        ? '#8AC0DD'
                        : data?.sponsor_party == 'Liberal'
                        ? '#FF9999'
                        : data?.sponsor_party == 'Green'
                        ? '#A9FA98'
                        : data?.sponsor_party == 'Conservative'
                        ? '#AECBF4'
                        : data?.sponsor_party == 'NDP'
                        ? '#F7B86D'
                        : '#DD3434',
                  }}
                >
                  {data?.sponsor_party}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-[100px_1fr] md:grid-cols-[120px_1fr] gap-2 md:gap-4">
              <h3 className="text-[#000] font-['SFProTextMedium'] text-sm md:text-base">
                Status
              </h3>
              <p className="text-[#515151] text-sm md:text-base">
                {data?.status}
              </p>
            </div>

            <div className="grid grid-cols-[100px_1fr] md:grid-cols-[120px_1fr] gap-2 md:gap-4">
              <h3 className="text-[#000] font-['SFProTextMedium'] text-sm md:text-base">
                Summary
              </h3>
              <p className="text-[#515151] text-sm md:text-base">
                {data?.summary}
              </p>
            </div>

            <div className="grid grid-cols-[100px_1fr] md:grid-cols-[120px_1fr] gap-2 md:gap-4">
              <h3 className="text-[#000] font-['SFProTextMedium'] text-sm md:text-base">
                Elsewhere
              </h3>
              <p className="text-[#515151] text-sm md:text-base">
                All sorts of information on this bill is available at{' '}
                <a
                  href={data?.legisinfo_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#007BFF] hover:underline"
                >
                  LEGISinfo
                </a>
                , an excellent resource from the Library of Parliament. You can
                also read the full text of the bill.
                <a
                  href={data?.text_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#007BFF] hover:underline"
                >
                  Here
                </a>
              </p>
            </div>

            <div className="grid grid-cols-[100px_1fr] md:grid-cols-[120px_1fr] gap-2 md:gap-4">
              <h3 className="text-[#000] font-['SFProTextMedium'] text-sm md:text-base">
                Votes
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-10">
                {data?.votes.map((vote, index) => (
                  <p
                    key={index}
                    className="text-[#515151] text-sm md:text-base"
                  >
                    {vote.date}{' '}
                    <span
                      className={`${
                        vote.result === 'Passed'
                          ? 'text-green-500'
                          : vote.result === 'Failed'
                          ? 'text-red-500'
                          : ''
                      }`}
                    >
                      {vote.result}
                    </span>{' '}
                    {vote.description}
                  </p>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div>
        <h1 className="text-[20px] md:text-[24px] font-[SFProTextBold] mt-12 md:mt-20 mb-6 md:mb-8">
          House Mentions
        </h1>
        <div className="space-y-4 md:space-y-6">
          {houseMentions.length
            ? houseMentions.map((mention, index) => (
                <DebateCard key={index} debate={mention} index={index} />
              ))
            : Array.from({ length: 3 }).map((_, i) => (
                <DebateCardSkeleton key={i} />
              ))}
        </div>
      </div>
    </div>
  );
};

export default Bills;
