from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json

# Router for Hidden Markov Model endpoints
HiddenMarkovRouter = APIRouter()

# Pydantic models for request and response
class HMMInput(BaseModel):
    transition_matrix: List[List[float]]  # A matrix: transitions between hidden states
    emission_matrix: List[List[float]]    # B matrix: emission probabilities
    initial_probabilities: List[float]    # π vector: initial state distribution
    hidden_states: List[str]              # Names of hidden states (e.g., "True", "Fake")
    observation_symbols: List[str]        # Names of observation symbols
    observations: List[int]               # Sequence of observations (indices into observation_symbols)

class HMMResult(BaseModel):
    steady_state: Dict[str, float]
    observation_likelihood: float
    most_likely_path: List[str]
    state_probabilities: List[Dict[str, float]]
    path_diagram: str  # Base64 encoded image

# Helper functions for HMM calculations
def forward_algorithm(A: np.ndarray, B: np.ndarray, pi: np.ndarray, observations: List[int]) -> tuple:
    """
    Implement the Forward algorithm for HMM.
    
    Args:
        A: Transition matrix (N x N)
        B: Emission matrix (N x M)
        pi: Initial state distribution (N)
        observations: Sequence of observation indices
        
    Returns:
        alpha: Forward variables
        likelihood: P(O|λ)
    """
    N = A.shape[0]  # Number of states
    T = len(observations)  # Length of observation sequence
    
    # Initialize forward variables
    alpha = np.zeros((T, N))
    
    # Initialization step
    alpha[0, :] = pi * B[:, observations[0]]
    
    # Recursion step
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, observations[t]]
    
    # Termination step
    likelihood = np.sum(alpha[T-1, :])
    
    return alpha, likelihood

def viterbi_algorithm(A: np.ndarray, B: np.ndarray, pi: np.ndarray, observations: List[int]) -> tuple:
    """
    Implement the Viterbi algorithm for HMM.
    
    Args:
        A: Transition matrix (N x N)
        B: Emission matrix (N x M)
        pi: Initial state distribution (N)
        observations: Sequence of observation indices
        
    Returns:
        best_path: Most likely sequence of hidden states
        max_prob: Probability of the best path
    """
    N = A.shape[0]  # Number of states
    T = len(observations)  # Length of observation sequence
    
    # Initialize variables
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization step
    delta[0, :] = pi * B[:, observations[0]]
    psi[0, :] = 0
    
    # Recursion step
    for t in range(1, T):
        for j in range(N):
            delta[t, j] = np.max(delta[t-1, :] * A[:, j]) * B[j, observations[t]]
            psi[t, j] = np.argmax(delta[t-1, :] * A[:, j])
    
    # Termination step
    best_path = np.zeros(T, dtype=int)
    max_prob = np.max(delta[T-1, :])
    best_path[T-1] = np.argmax(delta[T-1, :])
    
    # Path backtracking
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]
    
    return best_path, max_prob

def calculate_hmm_steady_state(A: np.ndarray) -> np.ndarray:
    """Calculate the steady-state probabilities of the hidden states."""
    n = A.shape[0]
    
    # Method: Eigenvalue decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.where(np.isclose(eigenvalues, 1.0))[0][0]
        steady_state = np.real(eigenvectors[:, idx] / np.sum(eigenvectors[:, idx]))
        return steady_state
    except:
        # Fallback: Power iteration
        pi = np.ones(n) / n
        for _ in range(1000):
            pi_new = pi @ A
            if np.allclose(pi, pi_new, rtol=1e-8):
                break
            pi = pi_new
        return pi

def generate_path_diagram(observations: List[int], best_path: List[int], 
                         hidden_states: List[str], observation_symbols: List[str],
                         state_probs: np.ndarray) -> str:
    """Generate a diagram showing the most likely path and state probabilities."""
    T = len(observations)
    N = len(hidden_states)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot the observations
    ax1.plot(range(T), observations, 'o-', label='Observations')
    ax1.set_ylabel('Observation Index')
    ax1.set_title('Observation Sequence')
    ax1.set_xticks(range(T))
    ax1.set_xticklabels([observation_symbols[obs] for obs in observations])
    ax1.grid(True)
    
    # Plot the most likely path
    ax2.plot(range(T), best_path, 'ro-', linewidth=2, label='Most Likely Path')
    ax2.set_ylabel('Hidden State')
    ax2.set_title('Viterbi Path and State Probabilities')
    ax2.set_xticks(range(T))
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(hidden_states)
    ax2.grid(True)
    
    # Plot state probabilities as heatmap
    im = ax2.imshow(state_probs.T, aspect='auto', cmap='viridis', alpha=0.6, 
                   extent=(-0.5, T-0.5, -0.5, N-0.5))
    fig.colorbar(im, ax=ax2, label='Probability')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode the image as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

@HiddenMarkovRouter.post("/analyze", response_model=HMMResult)
async def analyze_hmm(input_data: HMMInput):
    """Analyze a Hidden Markov Model and return various metrics."""
    try:
        # Convert input to numpy arrays
        A = np.array(input_data.transition_matrix)
        B = np.array(input_data.emission_matrix)
        pi = np.array(input_data.initial_probabilities)
        observations = input_data.observations
        hidden_states = input_data.hidden_states
        observation_symbols = input_data.observation_symbols
        
        # Validate input
        if A.shape[0] != A.shape[1]:
            raise HTTPException(status_code=400, detail="Transition matrix must be square")
        
        if A.shape[0] != len(hidden_states):
            raise HTTPException(status_code=400, detail="Number of hidden states must match transition matrix dimensions")
        
        if B.shape[0] != len(hidden_states) or B.shape[1] != len(observation_symbols):
            raise HTTPException(status_code=400, detail="Emission matrix dimensions must match number of states and observation symbols")
        
        if len(pi) != len(hidden_states):
            raise HTTPException(status_code=400, detail="Initial probabilities length must match number of states")
        
        # Check if rows sum to 1
        if not np.allclose(np.sum(A, axis=1), 1.0, rtol=1e-5) or not np.allclose(np.sum(B, axis=1), 1.0, rtol=1e-5):
            raise HTTPException(status_code=400, detail="Rows of transition and emission matrices must sum to 1")
        
        if not np.isclose(np.sum(pi), 1.0, rtol=1e-5):
            raise HTTPException(status_code=400, detail="Initial probabilities must sum to 1")
        
        # Calculate metrics
        alpha, likelihood = forward_algorithm(A, B, pi, observations)
        best_path, _ = viterbi_algorithm(A, B, pi, observations)
        steady_state = calculate_hmm_steady_state(A)
        
        # Calculate state probabilities at each time step (normalized alpha)
        state_probs = alpha / np.sum(alpha, axis=1, keepdims=True)
        
        # Generate path diagram
        path_diagram = generate_path_diagram(observations, best_path, hidden_states, 
                                           observation_symbols, state_probs)
        
        # Prepare response
        steady_state_dict = {state: float(prob) for state, prob in zip(hidden_states, steady_state)}
        most_likely_path = [hidden_states[state] for state in best_path]
        
        # State probabilities at each time step
        state_probabilities = []
        for t in range(len(observations)):
            state_probabilities.append({
                state: float(prob) for state, prob in zip(hidden_states, state_probs[t])
            })
        
        return HMMResult(
            steady_state=steady_state_dict,
            observation_likelihood=float(likelihood),
            most_likely_path=most_likely_path,
            state_probabilities=state_probabilities,
            path_diagram=path_diagram
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing HMM: {str(e)}")

@HiddenMarkovRouter.post("/upload")
async def upload_hmm_data(file: UploadFile = File(...)):
    """Upload a JSON file with HMM data."""
    try:
        content = await file.read()
        data = json.loads(content)
        
        # Validate the uploaded data
        required_fields = ["transition_matrix", "emission_matrix", "initial_probabilities", 
                          "hidden_states", "observation_symbols", "observations"]
        
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, 
                                   detail=f"Invalid data format. Missing required field: {field}")
        
        # Process the data
        input_data = HMMInput(
            transition_matrix=data["transition_matrix"],
            emission_matrix=data["emission_matrix"],
            initial_probabilities=data["initial_probabilities"],
            hidden_states=data["hidden_states"],
            observation_symbols=data["observation_symbols"],
            observations=data["observations"]
        )
        
        return await analyze_hmm(input_data)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
