import React, { useEffect, useState } from 'react';
import api from '../../api/axios';

 type EntityType = 'representatives' | 'bills' | 'committees' | 'votes' | 'events';

 type ApiList<T> = {
	 items: T[];
	 count: number;
	 limit: number;
	 offset: number;
 };

 const AdminEntities: React.FC = () => {
	 const [type, setType] = useState<EntityType>('representatives');
	 const [items, setItems] = useState<any[]>([]);
	 const [limit, setLimit] = useState<number>(25);
	 const [offset, setOffset] = useState<number>(0);
	 const [q, setQ] = useState<string>('');
	 const [loading, setLoading] = useState<boolean>(false);
	 const [error, setError] = useState<string | null>(null);

	 const fetchData = async () => {
		 setLoading(true);
		 setError(null);
		 try {
			 const res = await api.get(`/api/v1/entities/${type}`, { params: { limit, offset, q: q || undefined } });
			 const data = res.data as ApiList<any>;
			 setItems(data.items || []);
		 } catch (e: any) {
			 setError(e?.message || 'Failed to load entities');
		 } finally {
			 setLoading(false);
		 }
	 };

	 useEffect(() => {
		 fetchData();
		 // eslint-disable-next-line react-hooks/exhaustive-deps
	 }, [type, limit, offset]);

	 const nextPage = () => setOffset(offset + limit);
	 const prevPage = () => setOffset(Math.max(0, offset - limit));

	 const columns = () => {
		 switch (type) {
			 case 'representatives':
				 return ['id', 'name', 'party', 'district', 'email', 'phone'];
			 case 'bills':
				 return ['id', 'title', 'classification', 'session'];
			 case 'committees':
				 return ['id', 'name', 'classification'];
			 case 'votes':
				 return ['id', 'bill_id', 'member', 'vote'];
			 case 'events':
				 return ['id', 'title', 'date', 'text'];
			 default:
				 return [];
		 }
	 };

	 return (
		 <div className="min-h-screen bg-gray-100 p-8 space-y-6">
			 <h1 className="text-2xl font-bold">Entities</h1>
			 <div className="bg-white rounded-lg shadow p-4 flex items-center gap-3 flex-wrap">
				 <label className="text-sm">Type</label>
				 <select className="border rounded px-2 py-1" value={type} onChange={(e) => { setOffset(0); setType(e.target.value as EntityType); }}>
					 <option value="representatives">Representatives</option>
					 <option value="bills">Bills</option>
					 <option value="committees">Committees</option>
					 <option value="votes">Votes</option>
					 <option value="events">Events</option>
				 </select>
				 <label className="text-sm">Page size</label>
				 <select className="border rounded px-2 py-1" value={limit} onChange={(e) => { setOffset(0); setLimit(parseInt(e.target.value)); }}>
					 <option value={10}>10</option>
					 <option value={25}>25</option>
					 <option value={50}>50</option>
				 </select>
				 <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search..." className="border rounded px-2 py-1 flex-1 min-w-[200px]" />
				 <button className="bg-gray-200 px-3 py-1 rounded" onClick={() => { setOffset(0); fetchData(); }}>Search</button>
			 </div>
			 {error && <div className="text-red-600">{error}</div>}
			 <div className="bg-white rounded-lg shadow p-4">
				 {loading ? (
					 <div>Loading…</div>
				 ) : (
					 <div className="overflow-auto">
						 <table className="min-w-full text-sm">
							 <thead>
								 <tr className="text-left border-b">
									 {columns().map((h) => (
										 <th key={h} className="py-2 pr-4">{h}</th>
									 ))}
								 </tr>
							 </thead>
							 <tbody>
								 {items.map((row, idx) => (
									 <tr key={idx} className="border-b">
										 {columns().map((h) => (
											 <td key={h} className="py-2 pr-4">{String(row?.[h] ?? '')}</td>
										 ))}
									 </tr>
								 ))}
							 </tbody>
						 </table>
					 </div>
				 )}
			 </div>
			 <div className="flex items-center gap-3">
				 <button disabled={offset === 0} onClick={prevPage} className="bg-gray-200 px-3 py-1 rounded">Prev</button>
				 <div className="text-sm">Offset: {offset}</div>
				 <button onClick={nextPage} className="bg-gray-200 px-3 py-1 rounded">Next</button>
			 </div>
		 </div>
	 );
 };

 export default AdminEntities;