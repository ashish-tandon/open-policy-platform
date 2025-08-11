import { useEffect, useState } from 'react';
import { FaCircleInfo } from 'react-icons/fa6';
import { Link, useParams } from 'react-router-dom';
import api from '../../../api/axios';
import { MPCardProps } from '../../../components/MPCard';
import parse, { domToReact } from 'html-react-parser';

const MPProfile = () => {
  const { id } = useParams();
  const [mps, setMps] = useState<MPCardProps[]>([]);
  const [loading, setLoading] = useState(true); // loading state

  const options = {
    replace: (domNode: import('html-react-parser').DOMNode) => {
      if (domNode.type === 'tag' && domNode.name === 'a') {
        const { href } = domNode.attribs || {};
        if (href && href.startsWith('/')) {
          return (
            <Link
              to={href}
              className="text-[#628ECB] hover:text-[#4A6B99] transition-colors"
            >
              {domToReact(
                domNode.children as import('html-react-parser').DOMNode[],
                options,
              )}
            </Link>
          );
        }
      }
    },
  };

  const loadBills = async () => {
    try {
      const response = await api.get('/web/politician', {
        params: {
          id: id,
        },
      });

      setMps(response.data.politicians);
    } catch (error) {
      // showError(error.response?.data.message || error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBills();
  }, []);

  const getActivityStyle = (type: string) => {
    if (type.includes('Voted No')) return 'text-[#DD3434]';
    if (type.includes('Voted Yes')) return 'text-[#4F8C10]';
    if (type.includes('in the House')) return 'text-[#D9B88E]';
    if (type.includes('committee')) return 'text-[#A4DED2]';
    if (type.includes('legislation')) return 'text-[#E2B8E8]';
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center pt-8 animate-pulse">
        <div className="grid md:grid-cols-2 grid-cols-1 gap-4 md:gap-12 padding-x">
          <div className="h-10 bg-gray-200 rounded w-3/4"></div>
          <div className="h-20 bg-gray-200 rounded w-full"></div>
        </div>
        <div className="w-full max-w-[1200px] px-4 mt-20 flex gap-4 max-sm:flex-col shadow-md rounded-[24px] p-4">
          <div className="w-[152px] h-[152px] bg-gray-200 rounded-full"></div>
          <div className="flex-1 space-y-4">
            <div className="h-6 bg-gray-200 rounded w-1/2"></div>
            <div className="h-4 bg-gray-200 rounded w-full"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-4 bg-gray-200 rounded w-full"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center pt-8">
      <div className="grid md:grid-cols-2 grid-cols-1 gap-4 md:gap-12 padding-x">
        <h1 className="text-[32px] md:text-[40px] font-['SFProDisplay']">
          Current Members of <span className="text-[#AFAFAF]">Parliament</span>
        </h1>
        <p className="text-[#222222] text-[14px] md:mt-0 mt-2">
          Looking for your MP? Enter your postal code in the search field at the
          upper right of the page. Looking for an ex-Member? See our list of{' '}
          <Link to="/former-mps" className="underline">
            former MPs
          </Link>{' '}
          (since 1994).
        </p>
      </div>

      <div className="w-full max-w-[1200px] px-4 flex items-start gap-4 mt-20 max-sm:flex-col shadow-md shadow-[#0000000D] rounded-[24px] p-4">
        <img
          src={mps[0]?.politician_image}
          alt={mps[0]?.name}
          className="w-[120px] h-[120px] md:w-[152px] md:h-[152px] object-contain md:self-star"
          onError={(e) => {
            const img = e.target as HTMLImageElement;
            if (
              img.src !==
              'https://media.istockphoto.com/id/2041572395/vector/blank-avatar-photo-placeholder-icon-vector-illustration.webp?b=1&s=612x612&w=0&k=20&c=88LuT9lqQ6gHAvy7aQfxnRs_iK6KpnE-8QHDw3YyAUU='
            ) {
              img.src =
                'https://media.istockphoto.com/id/2041572395/vector/blank-avatar-photo-placeholder-icon-vector-illustration.webp?b=1&s=612x612&w=0&k=20&c=88LuT9lqQ6gHAvy7aQfxnRs_iK6KpnE-8QHDw3YyAUU=';
            }
          }}
        />

        <div>
          <div className="w-full">
            <p className="text-gray-600 mb-2">
              <span className="font-['SFProTextBold'] pr-2 text-black text-[16px]">
                {mps[0]?.name}
              </span>
              {mps[0]?.role} ({mps[0]?.province_name})
            </p>
            <p className="text-gray-600 mb-4">{mps[0]?.election_summary}</p>
          </div>

          <button className="flex items-center gap-2 text-[#515151] hover:text-[#4A6B99] transition-colors cursor-pointer">
            <FaCircleInfo color="#000" size={20} />
            Contact {mps[0]?.name}
          </button>

          {mps[0]?.activity.map((activity, index) => (
            <div key={index} className="py-2">
              {activity.isTitle ? (
                <p className="text-[#515151] font-['SFProTextMedium'] mb-1 border-b border-t border-[#EDEDED] pb-2 pt-2">
                  {activity.title}
                </p>
              ) : (
                <div className="flex flex-col gap-2">
                  <div className="max-sm:flex-col gap-2">
                    <span
                      className={`text-sm font-['SFProTextMedium'] whitespace-nowrap ${getActivityStyle(
                        activity.title,
                      )}`}
                    >
                      {activity.title}{' '}
                    </span>
                    <span className="text-sm text-[#515151]">
                      {parse(activity.text, options)}
                    </span>
                  </div>
                  {activity.subtitle && (
                    <div className="pl-6 ml-2 border-l-2 border-[#EDEDED]">
                      <p className="text-sm text-[#515151]">
                        {activity.subtitle}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MPProfile;
