import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import api from '../../../api/axios';

interface VoteResult {
  politician_name: string;
  party: string;
  vote: 'Yes' | 'No' | 'No Vote';
}

type PartyFilter =
  | 'All'
  | 'Liberal'
  | 'Conservative'
  | 'Independent'
  | 'NDP'
  | 'Bloc'
  | 'Green';

interface VoteData {
  vote_no: string;
  date: string;
  total_yes: number;
  total_no: number;
  yes_party: PartyFilter[];
  no_party: PartyFilter[];
  summary: Record<string, VoteResult[]>;
}

const VotePage = () => {
  const location = useLocation();
  const fullPath = location.pathname;

  const [selectedParty, setSelectedParty] = useState<PartyFilter>('All');
  const [data, setData] = useState<Partial<VoteData>>({});
  const [voteResults, setVoteResults] = useState<Record<string, VoteResult[]>>(
    {},
  );
  const [loading, setLoading] = useState(true);

  const getVoteColor = (vote: VoteResult['vote']) => {
    switch (vote) {
      case 'Yes':
        return 'text-green-600';
      case 'No':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const loadHouseMentions = async () => {
    try {
      setLoading(true);
      const response = await api.get('/web/get-votes/', {
        params: { vote: fullPath },
      });

      if (response.data?.success) {
        setData(response.data.data);
        setVoteResults(response.data.data.summary);
      }
    } catch (error) {
      console.error('Failed to load vote data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHouseMentions();
  }, []);

  interface FilterButtonProps {
    party: PartyFilter;
    count?: number;
    isYesVote?: boolean;
  }

  const FilterButton = ({
    party,
    count,
    isYesVote = true,
  }: FilterButtonProps) => {
    const isSelected = selectedParty === party;
    return (
      <button
        onClick={() => setSelectedParty(party)}
        className={`
          px-3 py-1.5 rounded-full text-sm transition-all
          ${
            isSelected
              ? isYesVote
                ? 'bg-green-100 text-green-700'
                : 'bg-red-100 text-red-700'
              : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
          }
        `}
      >
        {party} {count ? `(${count})` : ''}
      </button>
    );
  };

  const SkeletonFilterButton = () => (
    <div className="h-8 w-24 rounded-full bg-gray-200 animate-pulse" />
  );

  const SkeletonCard = () => (
    <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100 animate-pulse">
      <div className="flex items-start justify-between">
        <div>
          <div className="h-4 w-32 bg-gray-200 rounded mb-2"></div>
          <div className="h-3 w-20 bg-gray-200 rounded"></div>
        </div>
        <div className="h-4 w-10 bg-gray-200 rounded"></div>
      </div>
    </div>
  );

  const filteredResults =
    selectedParty === 'All'
      ? voteResults
      : Object.fromEntries(
          Object.entries(voteResults).filter(
            ([party]) => party === selectedParty,
          ),
        );

  return (
    <div className="pt-[2rem] pb-8">
      <div className="padding-x">
        {loading ? (
          <>
            <div className="h-10 w-48 bg-gray-200 rounded mb-2 animate-pulse"></div>
            <div className="h-4 w-32 bg-gray-200 rounded mb-6 animate-pulse"></div>

            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex items-center gap-2">
                <span className="text-lg font-medium">Results:</span>
                <div className="flex gap-1 flex-wrap">
                  {[...Array(3)].map((_, i) => (
                    <SkeletonFilterButton key={`yes-skeleton-${i}`} />
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex gap-1 flex-wrap">
                  {[...Array(3)].map((_, i) => (
                    <SkeletonFilterButton key={`no-skeleton-${i}`} />
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-8 space-y-8">
              {[1, 2, 3].map((_, groupIndex) => (
                <div key={groupIndex}>
                  <div className="h-6 w-48 bg-gray-200 rounded mb-4 animate-pulse"></div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[1, 2, 3].map((_, i) => (
                      <SkeletonCard key={i} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <>
            <h1 className="text-[32px] md:text-[40px] text-[#222222] font-['SFProDisplay']">
              Vote #{data.vote_no}
            </h1>
            <p className="text-[#AFAFAF] text-lg mt-1">on {data.date}</p>

            <div className="mt-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-medium">Results:</span>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-green-600 font-medium whitespace-nowrap">
                      Yes ({data.total_yes ?? 0})
                    </span>
                    <div className="flex gap-1 flex-wrap">
                      {data.yes_party?.map((vote, index) => (
                        <FilterButton
                          key={`yes-${vote}-${index}`}
                          party={vote}
                          isYesVote={true}
                        />
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-red-600 font-medium whitespace-nowrap">
                    No ({data.total_no ?? 0})
                  </span>
                  <div className="flex gap-1 flex-wrap">
                    {data.no_party?.map((vote, index) => (
                      <FilterButton
                        key={`no-${vote}-${index}`}
                        party={vote}
                        isYesVote={false}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 space-y-8">
              {Object.entries(filteredResults).map(([party, results]) => (
                <div key={party}>
                  <h2 className="text-[24px] mb-4">
                    {party} ({results.length})
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {results.map((result, index) => (
                      <div
                        key={index}
                        className="bg-white rounded-lg p-4 shadow-sm border border-gray-100"
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <h3 className="font-medium">
                              {result.politician_name}
                            </h3>
                            <p className="text-sm text-gray-500">
                              {result.party}
                            </p>
                          </div>
                          <span
                            className={`font-medium ${getVoteColor(
                              result.vote,
                            )}`}
                          >
                            {result.vote}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default VotePage;
