from fastapi.testclient import TestClient
from src.main import app

def test_notify_accepts():
    c = TestClient(app)
    r = c.post('/notify?message=hello')
    assert r.status_code == 200
    assert r.json().get('status') == 'accepted'