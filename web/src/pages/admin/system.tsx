import React, { useEffect, useState } from 'react';

const AdminSystem: React.FC = () => {
  const [data, setData] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.VITE_API_URL || 'http://localhost:9000';
    fetch(`${base}/api/status`)
      .then((r) => r.json())
      .then((j) => setData(j))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading…</div>;
  if (error) return <div className="p-8 text-red-600">Error: {error}</div>;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Services Status</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(data).map(([name, st]: any) => (
          <div key={name} className="border rounded p-4">
            <div className="font-semibold">{name}</div>
            <div className={st.status === 'ok' ? 'text-green-600' : 'text-red-600'}>
              {st.status}
            </div>
            <div className="text-sm text-gray-600">Target: {st.target}</div>
            {st.error && <div className="text-sm text-red-600">Error: {st.error}</div>}
            {st.http && <div className="text-sm">HTTP: {st.http}</div>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AdminSystem;
