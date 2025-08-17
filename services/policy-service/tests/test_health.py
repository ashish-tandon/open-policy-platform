from fastapi.testclient import TestClient
from src.main import app

def test_healthz():
    c = TestClient(app)
    r = c.get('/healthz')
    assert r.status_code == 200

def test_policies():
    c = TestClient(app)
    r = c.get('/policies')
    assert r.status_code == 200
    assert 'policies' in r.json()