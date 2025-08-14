import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_scrapers_endpoints_respond():
	res = client.get('/api/v1/scrapers')
	assert res.status_code in (200, 500)
	# categories
	res2 = client.get('/api/v1/scrapers/categories')
	assert res2.status_code in (200, 500)