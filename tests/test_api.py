import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    
    def test_health_endpoint(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert 'ml_available' in data
        assert 'llm_available' in data
    
    def test_predict_missing_text(self, client):
        response = client.post('/predict', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_empty_text(self, client):
        response = client.post('/predict', json={'text': ''})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_valid_text(self, client):
        response = client.post('/predict', json={
            'text': 'This is a test article about current events.'
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'is_fake' in data
        assert 'confidence' in data
        assert 'label' in data
        assert 'method' in data
    
    def test_predict_ml_method(self, client):
        response = client.post('/predict', json={
            'text': 'Test article',
            'method': 'ml'
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['method'] == 'ml'
    
    def test_predict_ml_endpoint(self, client):
        response = client.post('/predict/ml', json={
            'text': 'Test article'
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['method'] == 'ml'
    
    def test_predict_llm_endpoint_no_key(self, client):
        response = client.post('/predict/llm', json={
            'text': 'Test article'
        })
        # Should either work (if API key set) or return 503
        assert response.status_code in [200, 503]
    
    def test_stats_endpoint(self, client):
        response = client.get('/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'ml_available' in data
        assert 'llm_available' in data
    
    def test_invalid_json(self, client):
        response = client.post('/predict', 
                              data='invalid json',
                              content_type='application/json')
        assert response.status_code == 400

