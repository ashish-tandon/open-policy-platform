from fastapi.testclient import TestClient
from src.main import app

def test_config_set_get():
    c = TestClient(app)
    r = c.post('/config/foo?value=bar')
    assert r.status_code == 200
    r = c.get('/config/foo')
    assert r.status_code == 200
    assert r.json().get('value') == 'bar'