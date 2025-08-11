// import { useLocation } from 'react-router-dom';
import EmptyState from '../../../components/common/EmptyState';
import { FiInbox } from 'react-icons/fi';

function VoteSearch() {
  // const location = useLocation();
  // const fullPath = location.pathname.replace('/votes/', ''); // "44-1/927"
  return (
    <EmptyState
      icon={FiInbox}
      message="No items founds"
      description="Try adjusting your search or filters to find what you're looking for."
    />
  );
}

export default VoteSearch;
