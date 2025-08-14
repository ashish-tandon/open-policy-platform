import React from 'react';

const AdminApiDocs: React.FC = () => {
	const src = `${import.meta.env.VITE_API_URL}/docs`;
	return (
		<div className="min-h-screen bg-gray-100 p-8">
			<h1 className="text-2xl font-bold mb-4">API Documentation</h1>
			<div className="bg-white rounded-lg shadow p-2">
				<iframe title="Swagger UI" src={src} className="w-full h-[80vh] border-0" />
			</div>
		</div>
	);
};

export default AdminApiDocs;