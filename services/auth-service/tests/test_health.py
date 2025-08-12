from fastapi.testclient import TestClient
from src.main import app

def test_healthz():
    c = TestClient(app)
    r = c.get('/healthz')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'

def test_readyz():
    c = TestClient(app)
    r = c.get('/readyz')
    assert r.status_code == 200