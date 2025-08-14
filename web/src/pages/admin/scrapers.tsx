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

type Summary = {
	timestamp: string;
	total_scrapers: number;
	successful: number;
	failed: number;
	success_rate: number;
	total_records: number;
} | null;

type Run = {
	id: number;
	category: string;
	start_time?: string | null;
	end_time?: string | null;
	status: string;
	records_collected: number;
};

type Attempt = {
	id: number;
	scraper_name: string;
	attempt_number: number;
	started_at?: string | null;
	finished_at?: string | null;
	status: string;
	error_message?: string | null;
};

const AdminScrapers: React.FC = () => {
	const [scrapers, setScrapers] = useState<Scraper[]>([]);
	const [categories, setCategories] = useState<CategoriesSummary>({});
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [runCategory, setRunCategory] = useState<string>('parliamentary');
	const [running, setRunning] = useState(false);
	const [message, setMessage] = useState<string | null>(null);
	const [summary, setSummary] = useState<Summary>(null);
	const [runs, setRuns] = useState<Run[]>([]);
	const [latestRunId, setLatestRunId] = useState<number | null>(null);
	const [attempts, setAttempts] = useState<Attempt[]>([]);

	const fetchLatestAttempts = async () => {
		try {
			const latest = await api.get('/api/v1/scrapers/runs/latest');
			const run = latest.data;
			if (run && run.id) {
				setLatestRunId(run.id);
				const at = await api.get(`/api/v1/scrapers/runs/${run.id}/attempts`);
				setAttempts((at.data?.attempts as Attempt[]) || []);
			}
		} catch {
			setAttempts([]);
		}
	};

	const refresh = async () => {
		const [statusRes, catRes, sumRes, runsRes] = await Promise.all([
			api.get('/api/v1/scrapers'),
			api.get('/api/v1/scrapers/categories'),
			api.get('/api/v1/scrapers/summary').catch(() => ({ data: null })),
			api.get('/api/v1/scrapers/runs?limit=20').catch(() => ({ data: [] })),
		]);
		setScrapers(statusRes.data.scrapers || []);
		setCategories(catRes.data.categories || {});
		setSummary(sumRes.data || null);
		setRuns(runsRes.data || []);
		await fetchLatestAttempts();
	};

	useEffect(() => {
		const fetchData = async () => {
			try {
				await refresh();
			} catch (e: any) {
				setError(e?.message || 'Failed to load scrapers');
			} finally {
				setLoading(false);
			}
		};
		fetchData();
	}, []);

	const handleRunCategory = async () => {
		setRunning(true);
		setMessage(null);
		try {
			await api.post(`/api/v1/scrapers/run/category/${runCategory}`, {
				scraper_id: '',
				category: runCategory,
				max_records: 5,
				force_run: false,
			});
			setMessage(`Triggered ${runCategory} scrapers`);
			setTimeout(refresh, 2000);
		} catch (e: any) {
			setMessage(e?.message || 'Failed to trigger');
		} finally {
			setRunning(false);
		}
	};

	const handleRunFull = async () => {
		setRunning(true);
		setMessage(null);
		try {
			await api.post(`/api/v1/scrapers/run/full/${runCategory}`, null, { params: { retries: 2, max_records: 10 } });
			setMessage(`Triggered full runner for ${runCategory}`);
			setTimeout(refresh, 3000);
		} catch (e: any) {
			setMessage(e?.message || 'Failed to trigger full runner');
		} finally {
			setRunning(false);
		}
	};

	const handleQueueFull = async () => {
		setRunning(true);
		setMessage(null);
		try {
			const res = await api.post(`/api/v1/scrapers/queue/full/${runCategory}`, null, { params: { retries: 2, max_records: 10 } });
			const tid = (res.data?.task_id as string) || '';
			setMessage(`Queued full runner for ${runCategory} (task ${tid})`);
			setTimeout(refresh, 3000);
		} catch (e: any) {
			setMessage(e?.message || 'Failed to queue full runner');
		} finally {
			setRunning(false);
		}
	};

	return (
		<div className="min-h-screen bg-gray-100 p-8">
			<h1 className="text-2xl font-bold mb-6">Scrapers Management</h1>
			{loading && <div>Loading…</div>}
			{error && <div className="text-red-600">{error}</div>}
			{!loading && !error && (
				<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
					<div className="bg-white rounded-lg shadow p-6 space-y-4">
						<h2 className="font-semibold">Actions</h2>
						<div className="flex items-center gap-2">
							<select className="border rounded px-2 py-1" value={runCategory} onChange={(e) => setRunCategory(e.target.value)}>
								<option value="parliamentary">Parliamentary</option>
								<option value="provincial">Provincial</option>
								<option value="municipal">Municipal</option>
								<option value="civic">Civic</option>
								<option value="update">Update</option>
							</select>
							<button disabled={running} onClick={handleRunCategory} className="bg-blue-600 text-white px-3 py-1 rounded">
								{running ? 'Running…' : 'Run Category'}
							</button>
							<button disabled={running} onClick={handleRunFull} className="bg-emerald-600 text-white px-3 py-1 rounded">
								{running ? 'Running…' : 'Run Full (Retries)'}
							</button>
							<button disabled={running} onClick={handleQueueFull} className="bg-purple-600 text-white px-3 py-1 rounded">
								{running ? 'Queuing…' : 'Queue Full (Workers)'}
							</button>
						</div>
						{message && <div className="text-sm text-gray-600">{message}</div>}
						<h2 className="font-semibold">Categories</h2>
						<ul className="text-sm space-y-1">
							{Object.entries(categories).map(([cat, s]) => (
								<li key={cat}>
									<span className="font-medium">{cat}</span>: {s.count} scrapers, {s.active} active, {s.success_rate}% success
								</li>
							))}
						</ul>
						{summary && (
							<div className="mt-4 text-sm">
								<h3 className="font-semibold mb-1">Latest Summary</h3>
								<div>Total: {summary.total_scrapers}, Success: {summary.successful}, Failed: {summary.failed}</div>
								<div>Success Rate: {summary.success_rate}% | Records: {summary.total_records}</div>
								<div>Timestamp: {summary.timestamp}</div>
							</div>
						)}
					</div>
					<div className="bg-white rounded-lg shadow p-6 md:col-span-2 space-y-6">
						<div>
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
						<div>
							<h2 className="font-semibold mb-2">Recent Runs</h2>
							<div className="overflow-auto">
								<table className="min-w-full text-sm">
									<thead>
										<tr className="text-left border-b">
											<th className="py-2 pr-4">Run ID</th>
											<th className="py-2 pr-4">Category</th>
											<th className="py-2 pr-4">Status</th>
											<th className="py-2 pr-4">Records</th>
											<th className="py-2 pr-4">Start</th>
											<th className="py-2 pr-4">End</th>
										</tr>
									</thead>
									<tbody>
										{runs.map((r) => (
											<tr key={r.id} className="border-b">
												<td className="py-2 pr-4">{r.id}</td>
												<td className="py-2 pr-4">{r.category}</td>
												<td className="py-2 pr-4">{r.status}</td>
												<td className="py-2 pr-4">{r.records_collected}</td>
												<td className="py-2 pr-4">{r.start_time || '-'}</td>
												<td className="py-2 pr-4">{r.end_time || '-'}</td>
											</tr>
										))}
									</tbody>
								</table>
							</div>
							{attempts.length > 0 && (
								<div className="mt-4">
									<h3 className="font-semibold mb-1">Latest Run Attempts (Run {latestRunId})</h3>
									<div className="overflow-auto">
										<table className="min-w-full text-sm">
											<thead>
												<tr className="text-left border-b">
													<th className="py-2 pr-4">Scraper</th>
													<th className="py-2 pr-4">Attempt</th>
													<th className="py-2 pr-4">Status</th>
													<th className="py-2 pr-4">Started</th>
													<th className="py-2 pr-4">Finished</th>
													<th className="py-2 pr-4">Error</th>
												</tr>
											</thead>
											<tbody>
												{attempts.map((a) => (
													<tr key={a.id} className="border-b">
														<td className="py-2 pr-4">{a.scraper_name}</td>
														<td className="py-2 pr-4">{a.attempt_number}</td>
														<td className="py-2 pr-4">{a.status}</td>
														<td className="py-2 pr-4">{a.started_at || '-'}</td>
														<td className="py-2 pr-4">{a.finished_at || '-'}</td>
														<td className="py-2 pr-4">{a.error_message || '-'}</td>
													</tr>
												))}
											</tbody>
										</table>
									</div>
								</div>
							)}
						</div>
					</div>
			)}
		</div>
	);
};

export default AdminScrapers;
