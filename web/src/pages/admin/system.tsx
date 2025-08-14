import React, { useEffect, useState } from 'react';
import api from '../../api/axios';

type Health = { status: string };

type ScraperHealth = {
	status: string;
	total_scrapers: number;
	active_scrapers: number;
	success_rate: number;
	last_run?: string;
};

const AdminSystem: React.FC = () => {
	const [apiHealth, setApiHealth] = useState<Health | null>(null);
	const [scraperHealth, setScraperHealth] = useState<ScraperHealth | null>(null);
	const [metrics, setMetrics] = useState<string>('');
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const load = async () => {
			try {
				const [h, s] = await Promise.all([
					api.get('/api/v1/health'),
					api.get('/health/scrapers'),
				]);
				setApiHealth(h.data);
				setScraperHealth(s.data);
			} catch (e: any) {
				setError(e?.message || 'Failed to load system status');
			}
		};
		load();
	}, []);

	useEffect(() => {
		const fetchMetrics = async () => {
			try {
				const res = await api.get('/metrics', { responseType: 'text' });
				setMetrics(res.data as string);
			} catch {}
		};
		fetchMetrics();
	}, []);

	const parseMetric = (name: string): string | null => {
		const line = metrics.split('\n').find((l) => l.startsWith(name));
		if (!line) return null;
		const parts = line.trim().split(' ');
		return parts[parts.length - 1] || null;
	};

	const lastRunEpoch = parseMetric('openpolicy_scraper_latest_run_timestamp_seconds');
	const lastRun = lastRunEpoch ? new Date(parseInt(lastRunEpoch, 10) * 1000).toISOString() : '-';
	const runsCompleted = parseMetric('openpolicy_scraper_runs_total{status="completed"}') || '0';
	const runsFailed = parseMetric('openpolicy_scraper_runs_total{status="failed"}') || '0';

	const flowerUrl = 'http://localhost:5555';
	return (
		<div className="min-h-screen bg-gray-100 p-8 space-y-6">
			<h1 className="text-2xl font-bold">System Management</h1>
			{error && <div className="text-red-600">{error}</div>}
			<div className="grid grid-cols-1 md:grid-cols-4 gap-6">
				<div className="bg-white rounded-lg shadow p-6">
					<h2 className="font-semibold mb-2">API</h2>
					<div>Status: {apiHealth?.status ?? 'unknown'}</div>
				</div>
				<div className="bg-white rounded-lg shadow p-6">
					<h2 className="font-semibold mb-2">Scrapers</h2>
					<div>Status: {scraperHealth?.status ?? 'unknown'}</div>
					<div>Total: {scraperHealth?.total_scrapers ?? 0}, Active: {scraperHealth?.active_scrapers ?? 0}</div>
					<div>Success: {scraperHealth?.success_rate ?? 0}%</div>
					<div>Last Run: {scraperHealth?.last_run ?? '-'}</div>
				</div>
				<div className="bg-white rounded-lg shadow p-6">
					<h2 className="font-semibold mb-2">Workers</h2>
					<a className="text-blue-600 underline" href={flowerUrl} target="_blank" rel="noreferrer">Open Flower (if running)</a>
				</div>
				<div className="bg-white rounded-lg shadow p-6">
					<h2 className="font-semibold mb-2">Metrics</h2>
					<div>Last Run (from metrics): {lastRun}</div>
					<div>Runs Completed: {runsCompleted}</div>
					<div>Runs Failed: {runsFailed}</div>
				</div>
			</div>
		</div>
	);
};

export default AdminSystem;
