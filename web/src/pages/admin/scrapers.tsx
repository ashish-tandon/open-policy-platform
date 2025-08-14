import React, { useEffect, useState } from 'react';
import api from '../../api/axios';

type Scraper = {
	name: string;
	category: string;
	status: string;
	last_run?: string;
	success_rate?: number;
	records_collected?: number;
	error_count?: number;
};

type CategoriesSummary = Record<string, { count: number; active: number; success_rate: number }>;

const AdminScrapers: React.FC = () => {
	const [scrapers, setScrapers] = useState<Scraper[]>([]);
	const [categories, setCategories] = useState<CategoriesSummary>({});
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const fetchData = async () => {
			try {
				const [statusRes, catRes] = await Promise.all([
					api.get('/api/v1/scrapers'),
					api.get('/api/v1/scrapers/categories'),
				]);
				setScrapers(statusRes.data.scrapers || []);
				setCategories(catRes.data.categories || {});
			} catch (e: any) {
				setError(e?.message || 'Failed to load scrapers');
			} finally {
				setLoading(false);
			}
		};
		fetchData();
	}, []);

	return (
		<div className="min-h-screen bg-gray-100 p-8">
			<h1 className="text-2xl font-bold mb-6">Scrapers Management</h1>
			{loading && <div>Loading…</div>}
			{error && <div className="text-red-600">{error}</div>}
			{!loading && !error && (
				<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
					<div className="bg-white rounded-lg shadow p-6">
						<h2 className="font-semibold mb-2">Categories</h2>
						<ul className="text-sm space-y-1">
							{Object.entries(categories).map(([cat, s]) => (
								<li key={cat}>
									<span className="font-medium">{cat}</span>: {s.count} scrapers, {s.active} active, {s.success_rate}% success
								</li>
							))}
						</ul>
					</div>
					<div className="bg-white rounded-lg shadow p-6 md:col-span-2">
						<h2 className="font-semibold mb-2">Scrapers</h2>
						<div className="overflow-auto">
							<table className="min-w-full text-sm">
								<thead>
									<tr className="text-left border-b">
										<th className="py-2 pr-4">Name</th>
										<th className="py-2 pr-4">Category</th>
										<th className="py-2 pr-4">Status</th>
										<th className="py-2 pr-4">Success</th>
										<th className="py-2 pr-4">Records</th>
										<th className="py-2 pr-4">Last Run</th>
									</tr>
								</thead>
								<tbody>
									{scrapers.map((s) => (
										<tr key={s.name} className="border-b">
											<td className="py-2 pr-4">{s.name}</td>
											<td className="py-2 pr-4">{s.category}</td>
											<td className="py-2 pr-4">{s.status}</td>
											<td className="py-2 pr-4">{s.success_rate ?? 0}%</td>
											<td className="py-2 pr-4">{s.records_collected ?? 0}</td>
											<td className="py-2 pr-4">{s.last_run || '-'}</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};

export default AdminScrapers;
