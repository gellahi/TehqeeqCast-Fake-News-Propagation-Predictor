import pytest
from fastapi.testclient import TestClient
import json
import os
import sys

# Add the parent directory to the path so we can import the main app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to TehqeeqCast API - Fake-News Propagation Predictor"}

def test_markov_chain_analyze():
    # Test data
    test_data = {
        "transition_matrix": [
            [0.3, 0.3, 0.2, 0.1, 0.1],
            [0.2, 0.2, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.4, 0.2, 0.1],
            [0.1, 0.1, 0.2, 0.5, 0.1],
            [0.2, 0.1, 0.1, 0.1, 0.5]
        ],
        "states": ["Twitter", "Facebook", "Instagram", "WhatsApp", "TikTok"]
    }
    
    response = client.post("/api/markov/analyze", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "steady_state" in result
    assert "mean_recurrence_times" in result
    assert "mean_first_passage_times" in result
    assert "transition_diagram" in result
    
    # Check that steady state probabilities sum to approximately 1
    steady_state_sum = sum(result["steady_state"].values())
    assert abs(steady_state_sum - 1.0) < 1e-5

def test_hmm_analyze():
    # Test data
    test_data = {
        "hidden_states": ["True", "Partially_True", "Fake"],
        "observation_symbols": ["Low_Engagement", "Medium_Engagement", "High_Engagement", "Viral"],
        "transition_matrix": [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6]
        ],
        "emission_matrix": [
            [0.6, 0.3, 0.1, 0.0],
            [0.2, 0.4, 0.3, 0.1],
            [0.1, 0.2, 0.3, 0.4]
        ],
        "initial_probabilities": [0.5, 0.3, 0.2],
        "observations": [0, 1, 1, 2, 3, 3, 2, 1, 0]
    }
    
    response = client.post("/api/hmm/analyze", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "steady_state" in result
    assert "observation_likelihood" in result
    assert "most_likely_path" in result
    assert "state_probabilities" in result
    assert "path_diagram" in result
    
    # Check that steady state probabilities sum to approximately 1
    steady_state_sum = sum(result["steady_state"].values())
    assert abs(steady_state_sum - 1.0) < 1e-5
    
    # Check that the most likely path has the correct length
    assert len(result["most_likely_path"]) == len(test_data["observations"])

def test_mm1_queue_analyze():
    # Test data
    test_data = {
        "arrival_rate": 5.0,
        "service_rate": 6.0,
        "time_points": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    
    response = client.post("/api/queue/analyze", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "utilization" in result
    assert "average_queue_length" in result
    assert "average_system_length" in result
    assert "average_wait_time" in result
    assert "average_system_time" in result
    assert "probability_idle" in result
    assert "stability" in result
    assert "queue_diagram" in result
    
    # Check that utilization is correct
    assert abs(result["utilization"] - (test_data["arrival_rate"] / test_data["service_rate"])) < 1e-5
    
    # Check that the system is stable
    assert result["stability"] == True

def test_mm1_queue_unstable():
    # Test data with arrival rate > service rate (unstable)
    test_data = {
        "arrival_rate": 7.0,
        "service_rate": 6.0
    }
    
    response = client.post("/api/queue/analyze", json=test_data)
    assert response.status_code == 200
    
    # Check that the system is unstable
    result = response.json()
    assert result["stability"] == False
    assert result["average_queue_length"] == float('inf')
    assert result["average_system_length"] == float('inf')
    assert result["average_wait_time"] == float('inf')
    assert result["average_system_time"] == float('inf')
