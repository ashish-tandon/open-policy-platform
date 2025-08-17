from fastapi.testclient import TestClient
from src.main import app

def test_search_empty():
    c = TestClient(app)
    r = c.get('/search?q=')
    assert r.status_code == 200
    j = r.json()
    assert 'results' in j and j['total'] == 0