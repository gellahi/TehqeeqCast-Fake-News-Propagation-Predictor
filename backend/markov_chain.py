from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import json
from scipy import linalg

# Router for Markov Chain endpoints
MarkovChainRouter = APIRouter()

# Pydantic models for request and response
class MarkovChainInput(BaseModel):
    transition_matrix: List[List[float]]
    states: List[str]
    
class MarkovChainResult(BaseModel):
    steady_state: Dict[str, float]
    mean_recurrence_times: Dict[str, float]
    mean_first_passage_times: Dict[str, Dict[str, float]]
    transition_diagram: str  # Base64 encoded image
    
# Helper functions for Markov Chain calculations
def calculate_steady_state(P: np.ndarray) -> np.ndarray:
    """Calculate the steady-state probabilities of a Markov chain."""
    n = P.shape[0]
    
    # Method 1: Eigenvalue decomposition
    try:
        eigenvalues, eigenvectors = linalg.eig(P.T)
        idx = np.where(np.isclose(eigenvalues, 1.0))[0][0]
        steady_state = np.real(eigenvectors[:, idx] / np.sum(eigenvectors[:, idx]))
        return steady_state
    except:
        # Method 2: Solving linear system
        A = np.append(P.T - np.eye(n), np.ones((1, n)), axis=0)
        b = np.append(np.zeros(n), [1.0], axis=0)
        try:
            steady_state = linalg.lstsq(A, b)[0]
            return steady_state
        except:
            # Method 3: Power iteration
            pi = np.ones(n) / n
            for _ in range(1000):
                pi_new = pi @ P
                if np.allclose(pi, pi_new, rtol=1e-8):
                    break
                pi = pi_new
            return pi

def calculate_mean_recurrence_times(steady_state: np.ndarray) -> np.ndarray:
    """Calculate the mean recurrence times from steady-state probabilities."""
    return 1.0 / steady_state

def calculate_mean_first_passage_times(P: np.ndarray, steady_state: np.ndarray) -> np.ndarray:
    """Calculate the mean first passage times between states."""
    n = P.shape[0]
    M = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0 / steady_state[i]
            else:
                # Create a modified transition matrix for absorption
                P_abs = P.copy()
                P_abs[j, :] = 0
                P_abs[j, j] = 1
                
                # Solve the system (I - P_abs)x = 1 for all states except j
                idx = np.arange(n) != j
                A = np.eye(n-1) - P_abs[np.ix_(idx, idx)]
                b = np.ones(n-1)
                
                try:
                    x = linalg.solve(A, b)
                    M[i, j] = x[np.where(idx)[0] == i][0] if i in np.where(idx)[0] else 0
                except:
                    M[i, j] = float('inf')  # If state j is not reachable from i
    
    return M

def generate_transition_diagram(P: np.ndarray, states: List[str]) -> str:
    """Generate a transition diagram for the Markov chain and return as base64 encoded image."""
    G = nx.DiGraph()
    
    # Add nodes
    for i, state in enumerate(states):
        G.add_node(state)
    
    # Add edges with weights
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if P[i, j] > 0:
                G.add_edge(from_state, to_state, weight=P[i, j], label=f"{P[i, j]:.2f}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=20)
    
    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode the image as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

@MarkovChainRouter.post("/analyze", response_model=MarkovChainResult)
async def analyze_markov_chain(input_data: MarkovChainInput):
    """Analyze a Markov chain and return various metrics."""
    try:
        # Convert input to numpy arrays
        P = np.array(input_data.transition_matrix)
        states = input_data.states
        
        # Validate input
        if P.shape[0] != P.shape[1]:
            raise HTTPException(status_code=400, detail="Transition matrix must be square")
        
        if P.shape[0] != len(states):
            raise HTTPException(status_code=400, detail="Number of states must match transition matrix dimensions")
        
        # Check if rows sum to 1
        row_sums = np.sum(P, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            raise HTTPException(status_code=400, detail="Rows of transition matrix must sum to 1")
        
        # Calculate metrics
        steady_state = calculate_steady_state(P)
        mean_recurrence_times = calculate_mean_recurrence_times(steady_state)
        mean_first_passage_times = calculate_mean_first_passage_times(P, steady_state)
        
        # Generate transition diagram
        transition_diagram = generate_transition_diagram(P, states)
        
        # Prepare response
        steady_state_dict = {state: float(prob) for state, prob in zip(states, steady_state)}
        recurrence_times_dict = {state: float(time) for state, time in zip(states, mean_recurrence_times)}
        
        passage_times_dict = {}
        for i, from_state in enumerate(states):
            passage_times_dict[from_state] = {}
            for j, to_state in enumerate(states):
                if i != j:  # Exclude recurrence times
                    passage_times_dict[from_state][to_state] = float(mean_first_passage_times[i, j])
        
        return MarkovChainResult(
            steady_state=steady_state_dict,
            mean_recurrence_times=recurrence_times_dict,
            mean_first_passage_times=passage_times_dict,
            transition_diagram=transition_diagram
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing Markov chain: {str(e)}")

@MarkovChainRouter.post("/upload")
async def upload_markov_data(file: UploadFile = File(...)):
    """Upload a JSON file with Markov chain data."""
    try:
        content = await file.read()
        data = json.loads(content)
        
        # Validate the uploaded data
        if "transition_matrix" not in data or "states" not in data:
            raise HTTPException(status_code=400, detail="Invalid data format. Must include 'transition_matrix' and 'states'")
        
        # Process the data (same as in the analyze endpoint)
        input_data = MarkovChainInput(
            transition_matrix=data["transition_matrix"],
            states=data["states"]
        )
        
        return await analyze_markov_chain(input_data)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
