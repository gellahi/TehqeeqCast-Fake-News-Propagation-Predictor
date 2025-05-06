# TehqeeqCast: Fake-News Propagation Predictor

## Team Members
1. Umair Altaf 22F3737
2. Anas Altaf 22F3639
3. Anas Ali 22F3673
4. Gohar Ellahi 22F3636

## 1. Detail of the Web-Based System

TehqeeqCast is a comprehensive web-based system designed to model and predict the propagation of fake news across social media platforms using stochastic processes. The system implements three key mathematical models:

1. **Markov Chain**: Models how misinformation transitions between different social media platforms
2. **Hidden Markov Model (HMM)**: Infers the hidden credibility state of posts based on observable engagement metrics
3. **M/M/1 Queue**: Forecasts moderation backlogs in fact-checking pipelines

### Architecture

The system follows a modern client-server architecture:

- **Backend**: Python with FastAPI framework, implementing mathematical models and data processing
- **Frontend**: Next.js with TypeScript, providing an intuitive user interface with interactive visualizations
- **Data Flow**: JSON-based communication between frontend and backend via RESTful API endpoints

### Key Features

- **Interactive Data Input**: Users can upload JSON files or manually input parameters for each model
- **Comprehensive Analysis**: Calculates key metrics such as steady-state probabilities, mean passage times, and queue statistics
- **Visual Representations**: Generates interactive charts and diagrams to visualize model results
- **Responsive Design**: Adapts to different screen sizes with a mobile-friendly interface
- **Dark-Themed UI**: Features a premium dark theme with amber accent colors and subtle animations

### User Interface

The user interface is designed with a clean, modern aesthetic featuring:

- Navigation bar for easy access to different models
- Home page with overview of available models
- Dedicated pages for each stochastic model
- File upload and manual input options
- Results display with interactive visualizations
- Responsive design for all device sizes

![TehqeeqCast Home Page](https://i.imgur.com/example1.png)
*Figure 1: TehqeeqCast home page showing the three main models*

![Markov Chain Analysis](https://i.imgur.com/example2.png)
*Figure 2: Markov Chain analysis page with transition diagram and steady-state probabilities*

## 2. Markov Models

### Problem Statement

In the context of fake news propagation, understanding how misinformation moves between different social media platforms is crucial for developing effective countermeasures. Traditional approaches often fail to capture the stochastic nature of content sharing across platforms.

### Solution Approach

We model the propagation of fake news across social media platforms as a Markov process, where:

- **States**: Different social media platforms (Twitter, Facebook, Instagram, WhatsApp, TikTok)
- **Transition Matrix**: Probabilities of content moving from one platform to another
- **Key Metrics**:
  - Steady-state distribution: Long-term proportion of fake news on each platform
  - Mean first passage times: Expected time for fake news to reach a specific platform
  - Mean recurrence times: Expected time for fake news to return to the same platform

### Mathematical Foundation

For a set of platforms S = {s₁, s₂, ..., sₙ}, we define a transition matrix P where P_{ij} represents the probability of content moving from platform i to platform j. The steady-state distribution π satisfies:

π·P = π and Σπᵢ = 1

### Implementation Highlights

The Markov Chain implementation calculates:
- Steady-state probabilities using eigenvalue decomposition
- Mean recurrence times from steady-state probabilities
- Mean first passage times using absorption probabilities
- Transition diagrams using NetworkX and Matplotlib

## 3. Hidden Markov Models

### Problem Statement

The true credibility of social media posts is not directly observable. However, we can observe engagement metrics like likes, shares, and comments. The challenge is to infer the hidden credibility state of posts based on these observable metrics.

### Solution Approach

We use a Hidden Markov Model to represent the evolution of a post's credibility:

- **Hidden States**: True credibility levels (True, Partially True, Fake)
- **Observations**: Engagement metrics (Low, Medium, High, Viral)
- **Key Algorithms**:
  - Forward algorithm: Calculates the likelihood of an observation sequence
  - Viterbi algorithm: Finds the most likely sequence of hidden states

### Mathematical Foundation

Our HMM is defined by:
- Hidden states Q = {q₁, q₂, ..., qₙ} (credibility levels)
- Observation symbols O = {o₁, o₂, ..., oₘ} (engagement levels)
- Transition probabilities A = {aᵢⱼ} where aᵢⱼ = P(qⱼ at t+1 | qᵢ at t)
- Emission probabilities B = {bᵢ(k)} where bᵢ(k) = P(oₖ | qᵢ)
- Initial state distribution π = {πᵢ}

### Implementation Highlights

The HMM implementation includes:
- Forward algorithm for calculating observation likelihoods
- Viterbi algorithm for finding the most likely credibility path
- State probability visualization over time
- Path diagrams showing the relationship between observations and hidden states

## 4. Queuing Theory

### Problem Statement

Content moderation teams face significant backlogs when reviewing flagged posts. Understanding queue dynamics helps platforms allocate resources efficiently to minimize the time that potentially harmful content remains visible.

### Solution Approach

We model the content moderation process as an M/M/1 queue:

- **Arrival Rate (λ)**: Rate at which posts are flagged for review
- **Service Rate (μ)**: Rate at which moderators process content
- **Key Metrics**:
  - Server utilization: Proportion of time moderators are busy
  - Average queue length: Expected number of posts waiting for review
  - Average wait time: Expected time a post spends waiting for review

### Mathematical Foundation

For an M/M/1 queue with arrival rate λ and service rate μ, key metrics include:
- Server utilization: ρ = λ/μ
- Average queue length: Lq = ρ²/(1-ρ)
- Average system length: L = ρ/(1-ρ)
- Average waiting time: Wq = Lq/λ
- Average system time: W = L/λ

### Implementation Highlights

The M/M/1 Queue implementation includes:
- Queue stability analysis
- Calculation of key performance metrics
- Visualization of state probabilities
- Time-series plots of queue behavior
- Server utilization analysis

## 5. Codes

### Programming Languages and Technologies

- **Backend**: Python 3.9 with FastAPI, NumPy, SciPy, NetworkX, Matplotlib
- **Frontend**: TypeScript with Next.js, React, Recharts, Styled Components

### Backend Code Snippets

#### FastAPI Main Application

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from markov_chain import MarkovChainRouter
from hidden_markov import HiddenMarkovRouter
from mm1_queue import MM1QueueRouter

app = FastAPI(
    title="TehqeeqCast API",
    description="Fake-News Propagation Predictor API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for each model
app.include_router(MarkovChainRouter, prefix="/api/markov", tags=["Markov Chain"])
app.include_router(HiddenMarkovRouter, prefix="/api/hmm", tags=["Hidden Markov Model"])
app.include_router(MM1QueueRouter, prefix="/api/queue", tags=["M/M/1 Queue"])

@app.get("/")
async def root():
    return {"message": "Welcome to TehqeeqCast API - Fake-News Propagation Predictor"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

#### Markov Chain Implementation

```python
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

# Calculate steady-state probabilities
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

# Calculate mean recurrence times
def calculate_mean_recurrence_times(steady_state: np.ndarray) -> np.ndarray:
    """Calculate the mean recurrence times from steady-state probabilities."""
    return 1.0 / steady_state

# Calculate mean first passage times
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

# Generate transition diagram
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
    nx.draw_networkx_nodes(G, pos, node_color='#FFC107', node_size=500, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=20, edge_color='#424242')

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='#000000')

    # Set background color
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().set_facecolor('#1e1e1e')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
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

#### Hidden Markov Model Implementation

```python
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

# Forward algorithm implementation
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

# Viterbi algorithm implementation
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

# Generate path diagram
def generate_path_diagram(observations: List[int], best_path: List[int],
                         hidden_states: List[str], observation_symbols: List[str],
                         state_probs: np.ndarray) -> str:
    """Generate a diagram showing the most likely path and state probabilities."""
    T = len(observations)
    N = len(hidden_states)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor('#1e1e1e')

    # Plot the observations
    ax1.plot(range(T), observations, 'o-', color='#FFC107', linewidth=2, label='Observations')
    ax1.set_ylabel('Observation Index', color='white')
    ax1.set_title('Observation Sequence', color='white')
    ax1.set_xticks(range(T))
    ax1.set_xticklabels([observation_symbols[obs] for obs in observations], color='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1e1e1e')

    # Plot the most likely path
    ax2.plot(range(T), best_path, 'ro-', color='#FFC107', linewidth=2, label='Most Likely Path')
    ax2.set_ylabel('Hidden State', color='white')
    ax2.set_title('Viterbi Path and State Probabilities', color='white')
    ax2.set_xticks(range(T))
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(hidden_states, color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#1e1e1e')

    # Plot state probabilities as heatmap
    im = ax2.imshow(state_probs.T, aspect='auto', cmap='viridis', alpha=0.6,
                   extent=(-0.5, T-0.5, -0.5, N-0.5))
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Probability', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
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

        # Calculate steady state of the transition matrix
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.where(np.isclose(eigenvalues, 1.0))[0][0]
        steady_state = np.real(eigenvectors[:, idx] / np.sum(eigenvectors[:, idx]))

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
```

#### M/M/1 Queue Implementation

```python
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json

# Router for M/M/1 Queue endpoints
MM1QueueRouter = APIRouter()

# Pydantic models for request and response
class MM1QueueInput(BaseModel):
    arrival_rate: float  # λ: average arrival rate
    service_rate: float  # μ: average service rate
    time_points: Optional[List[float]] = None  # Optional time points for plotting

class MM1QueueResult(BaseModel):
    utilization: float  # ρ = λ/μ
    average_queue_length: float  # Lq
    average_system_length: float  # L
    average_wait_time: float  # Wq
    average_system_time: float  # W
    queue_diagram: str  # Base64 encoded image
    probability_idle: float  # P0
    stability: bool  # Is the queue stable?

# Calculate M/M/1 queue metrics
def calculate_mm1_metrics(arrival_rate: float, service_rate: float) -> dict:
    """Calculate the standard metrics for an M/M/1 queue."""
    # Check stability condition
    if arrival_rate >= service_rate:
        return {
            "utilization": arrival_rate / service_rate,
            "average_queue_length": float('inf'),
            "average_system_length": float('inf'),
            "average_wait_time": float('inf'),
            "average_system_time": float('inf'),
            "probability_idle": 0,
            "stability": False
        }

    # Calculate metrics
    utilization = arrival_rate / service_rate
    probability_idle = 1 - utilization

    # Average number in queue (Lq)
    average_queue_length = (utilization**2) / (1 - utilization)

    # Average number in system (L)
    average_system_length = utilization / (1 - utilization)

    # Average waiting time in queue (Wq)
    average_wait_time = average_queue_length / arrival_rate

    # Average time in system (W)
    average_system_time = average_system_length / arrival_rate

    return {
        "utilization": utilization,
        "average_queue_length": average_queue_length,
        "average_system_length": average_system_length,
        "average_wait_time": average_wait_time,
        "average_system_time": average_system_time,
        "probability_idle": probability_idle,
        "stability": True
    }

# Generate queue diagram
def generate_queue_diagram(arrival_rate: float, service_rate: float, time_points: List[float] = None) -> str:
    """Generate diagrams for the M/M/1 queue metrics."""
    # If time_points not provided, create a default range
    if time_points is None or len(time_points) == 0:
        time_points = np.linspace(0, 10, 100)

    # Calculate metrics
    metrics = calculate_mm1_metrics(arrival_rate, service_rate)
    utilization = metrics["utilization"]

    # Create the plot with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1e1e1e')

    # 1. Plot state probabilities
    states = np.arange(0, 15)
    if metrics["stability"]:
        probabilities = [(1 - utilization) * (utilization ** n) for n in states]
    else:
        probabilities = [0 if n > 0 else 1 for n in states]

    axs[0, 0].bar(states, probabilities, color='#FFC107')
    axs[0, 0].set_title('State Probabilities P(n)', color='white')
    axs[0, 0].set_xlabel('Number of Customers (n)', color='white')
    axs[0, 0].set_ylabel('Probability', color='white')
    axs[0, 0].tick_params(axis='x', colors='white')
    axs[0, 0].tick_params(axis='y', colors='white')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_facecolor('#1e1e1e')

    # 2. Plot cumulative distribution function of waiting time
    if metrics["stability"]:
        wait_times = np.linspace(0, 5 * metrics["average_wait_time"], 100)
        cdf_values = 1 - np.exp(-service_rate * (1 - utilization) * wait_times)
    else:
        wait_times = np.linspace(0, 10, 100)
        cdf_values = np.zeros_like(wait_times)

    axs[0, 1].plot(wait_times, cdf_values, color='#FFC107', linewidth=2)
    axs[0, 1].set_title('CDF of Waiting Time', color='white')
    axs[0, 1].set_xlabel('Waiting Time', color='white')
    axs[0, 1].set_ylabel('Probability (Wq ≤ t)', color='white')
    axs[0, 1].tick_params(axis='x', colors='white')
    axs[0, 1].tick_params(axis='y', colors='white')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_facecolor('#1e1e1e')

    # 3. Plot average queue length over time (transient behavior)
    if metrics["stability"]:
        # Simple approximation of transient behavior
        steady_state = metrics["average_queue_length"]
        transient = [steady_state * (1 - np.exp(-service_rate * (1 - utilization) * t)) for t in time_points]
    else:
        # Unstable queue grows linearly
        transient = [(arrival_rate - service_rate) * t for t in time_points]

    axs[1, 0].plot(time_points, transient, color='#FFC107', linewidth=2)
    axs[1, 0].set_title('Average Queue Length Over Time', color='white')
    axs[1, 0].set_xlabel('Time', color='white')
    axs[1, 0].set_ylabel('Average Queue Length', color='white')
    axs[1, 0].tick_params(axis='x', colors='white')
    axs[1, 0].tick_params(axis='y', colors='white')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_facecolor('#1e1e1e')

    # 4. Plot server utilization vs arrival rate
    lambda_range = np.linspace(0, service_rate * 0.99, 100)
    utilizations = lambda_range / service_rate

    axs[1, 1].plot(lambda_range, utilizations, color='#FFC107', linewidth=2)
    axs[1, 1].axvline(x=arrival_rate, color='#F44336', linestyle='--', label=f'λ = {arrival_rate}')
    axs[1, 1].axhline(y=utilization, color='#4CAF50', linestyle='--', label=f'ρ = {utilization:.2f}')
    axs[1, 1].set_title('Server Utilization vs Arrival Rate', color='white')
    axs[1, 1].set_xlabel('Arrival Rate (λ)', color='white')
    axs[1, 1].set_ylabel('Server Utilization (ρ)', color='white')
    axs[1, 1].set_xlim(0, service_rate)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].legend(facecolor='#1e1e1e', edgecolor='#424242', labelcolor='white')
    axs[1, 1].tick_params(axis='x', colors='white')
    axs[1, 1].tick_params(axis='y', colors='white')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_facecolor('#1e1e1e')

    # Add overall title
    plt.suptitle(f'M/M/1 Queue Analysis (λ={arrival_rate}, μ={service_rate})',
                fontsize=16, color='white')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
    plt.close()

    # Encode the image as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_str}"

@MM1QueueRouter.post("/analyze", response_model=MM1QueueResult)
async def analyze_mm1_queue(input_data: MM1QueueInput):
    """Analyze an M/M/1 queue and return various metrics."""
    try:
        # Extract input parameters
        arrival_rate = input_data.arrival_rate
        service_rate = input_data.service_rate
        time_points = input_data.time_points

        # Validate input
        if arrival_rate <= 0 or service_rate <= 0:
            raise HTTPException(status_code=400, detail="Arrival and service rates must be positive")

        # Calculate metrics
        metrics = calculate_mm1_metrics(arrival_rate, service_rate)

        # Generate queue diagram
        queue_diagram = generate_queue_diagram(arrival_rate, service_rate, time_points)

        # Prepare response
        return MM1QueueResult(
            utilization=metrics["utilization"],
            average_queue_length=metrics["average_queue_length"],
            average_system_length=metrics["average_system_length"],
            average_wait_time=metrics["average_wait_time"],
            average_system_time=metrics["average_system_time"],
            probability_idle=metrics["probability_idle"],
            stability=metrics["stability"],
            queue_diagram=queue_diagram
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing M/M/1 queue: {str(e)}")
```

## 6. Frontend Implementation

### 6.1 Pages

#### Home Page (index.tsx)

```tsx
import React from 'react';
import styled from 'styled-components';
import Link from 'next/link';
import {
  Card,
  Button,
  Grid,
  SectionTitle,
  NeonText
} from '../components/ui/StyledComponents';

const HeroSection = styled.div`
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.xxl};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.xxl} 0;
  animation: fadeIn 1s ease-out;
`;

const HeroTitle = styled.h1`
  font-size: 4rem;
  margin-bottom: ${({ theme }) => theme.spacing.md};
  background: linear-gradient(to right, ${({ theme }) => theme.colors.primary}, #fff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: glow 2s ease-in-out infinite alternate;
`;

const HeroSubtitle = styled.p`
  font-size: 1.5rem;
  max-width: 800px;
  margin: 0 auto;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const FeaturesSection = styled.section`
  padding: ${({ theme }) => theme.spacing.xl} 0;
`;

const FeatureGrid = styled(Grid)`
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${({ theme }) => theme.spacing.xl};
`;

const FeatureCard = styled(Card)`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: ${({ theme }) => theme.spacing.xl};
  transition: transform 0.3s ease, box-shadow 0.3s ease;

  &:hover {
    transform: translateY(-10px);
    box-shadow: ${({ theme }) => theme.shadows.large};
  }

  h3 {
    margin: ${({ theme }) => theme.spacing.md} 0;
  }

  p {
    color: ${({ theme }) => theme.colors.textSecondary};
    margin-bottom: ${({ theme }) => theme.spacing.lg};
  }
`;

const FeatureIcon = styled.div`
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 50%;
  margin-bottom: ${({ theme }) => theme.spacing.md};
  position: relative;
  transition: all 0.3s ease;

  &:before {
    content: '';
    position: absolute;
    top: -5px;
    left: -5px;
    right: -5px;
    bottom: -5px;
    border: 1px solid ${({ theme }) => theme.colors.primary}20;
    border-radius: 50%;
    transition: all 0.3s ease;
  }

  svg {
    transform: scale(0.9);
    transition: transform 0.3s ease;
  }

  /* Enhanced hover effects */
  ${FeatureCard}:hover & {
    transform: translateY(-5px);
    background-color: rgba(0, 0, 0, 0.3);

    &:before {
      border-color: ${({ theme }) => theme.colors.primary}40;
    }

    svg {
      transform: scale(1);
    }
  }
`;

const PageContainer = styled.div`
  animation: fadeIn 0.5s ease-out;
  min-height: 100vh;
`;

const Home: React.FC = () => {
  return (
    <PageContainer>
      <HeroSection>
        <HeroTitle>TehqeeqCast</HeroTitle>
        <HeroSubtitle>
          A Stochastic Modeling Tool for Fake-News Propagation Prediction
        </HeroSubtitle>
        <Button
          variant="primary"
          as="a"
          href="/markov"
          style={{
            animation: 'slideUp 0.8s ease-out 0.4s both',
            marginTop: '20px'
          }}
        >
          Get Started
        </Button>
      </HeroSection>

      <FeaturesSection>
        <SectionTitle style={{ textAlign: 'center', marginBottom: '40px' }}>
          Explore Our Models
        </SectionTitle>

        <FeatureGrid>
          <FeatureCard>
            <FeatureIcon>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="6" cy="6" r="2" stroke="#FFC107" strokeWidth="1.5" />
                <circle cx="18" cy="6" r="2" stroke="#FFC107" strokeWidth="1.5" />
                <circle cx="12" cy="18" r="2" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="7.5" y1="6" x2="16.5" y2="6" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="6" y1="7.5" x2="12" y2="16.5" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="18" y1="7.5" x2="12" y2="16.5" stroke="#FFC107" strokeWidth="1.5" />
              </svg>
            </FeatureIcon>
            <h3><NeonText color="primary">Markov Chain</NeonText></h3>
            <p>
              Model how misinformation hops across social media platforms with transition probabilities,
              steady-state analysis, and passage time calculations.
            </p>
            <Link href="/markov" passHref>
              <Button variant="primary" style={{ marginTop: '10px' }}>Explore Markov Chain</Button>
            </Link>
          </FeatureCard>

          <FeatureCard>
            <FeatureIcon>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="6" width="18" height="12" rx="1" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="3" y1="10" x2="21" y2="10" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="3" y1="14" x2="21" y2="14" stroke="#FFC107" strokeWidth="1.5" />
                <circle cx="7" cy="8" r="1" fill="#FFC107" />
                <circle cx="12" cy="12" r="1" fill="#FFC107" />
                <circle cx="17" cy="16" r="1" fill="#FFC107" />
              </svg>
            </FeatureIcon>
            <h3><NeonText color="secondary">Hidden Markov Model</NeonText></h3>
            <p>
              Infer each post's hidden credibility trajectory from observable engagement metrics
              using forward and Viterbi algorithms.
            </p>
            <Link href="/hmm" passHref>
              <Button variant="primary" style={{ marginTop: '10px' }}>Explore HMM</Button>
            </Link>
          </FeatureCard>

          <FeatureCard>
            <FeatureIcon>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="9" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="12" y1="3" x2="12" y2="12" stroke="#FFC107" strokeWidth="1.5" />
                <line x1="12" y1="12" x2="18" y2="12" stroke="#FFC107" strokeWidth="1.5" />
                <circle cx="12" cy="12" r="1.2" fill="#FFC107" />
              </svg>
            </FeatureIcon>
            <h3><NeonText color="accent">M/M/1 Queue</NeonText></h3>
            <p>
              Forecast moderation backlogs with queueing theory, analyzing arrival rates,
              service times, and system stability for fact-checking pipelines.
            </p>
            <Link href="/queue" passHref>
              <Button variant="primary" style={{ marginTop: '10px' }}>Explore M/M/1 Queue</Button>
            </Link>
          </FeatureCard>
        </FeatureGrid>
      </FeaturesSection>
    </PageContainer>
  );
};

export default Home;
```

#### Markov Chain Page (markov.tsx)

```tsx
import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import {
  PageTitle,
  Card,
  Button,
  FormGroup,
  Label,
  Input,
  TextArea,
  Spinner,
  ErrorMessage
} from '../components/ui/StyledComponents';
import FileUploader from '../components/common/FileUploader';
import MarkovResults from '../components/results/MarkovResults';
import { markovApi } from '../services/api';

interface MarkovFormData {
  manualInput: string;
}

const MarkovPage: React.FC = () => {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inputMethod, setInputMethod] = useState<'file' | 'manual'>('file');

  const { register, handleSubmit, formState: { errors } } = useForm<MarkovFormData>();

  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true);
      setError(null);

      const data = await markovApi.upload(file);
      setResults(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred while processing the file');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleManualSubmit = async (data: MarkovFormData) => {
    try {
      setLoading(true);
      setError(null);

      // Parse the manual input as JSON
      const parsedData = JSON.parse(data.manualInput);
      const results = await markovApi.analyze(parsedData);
      setResults(results);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred while processing the input');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageTitle>Markov Chain Analysis</PageTitle>

      <Card>
        <div style={{ marginBottom: '20px' }}>
          <Button
            variant={inputMethod === 'file' ? 'primary' : undefined}
            onClick={() => setInputMethod('file')}
            style={{ marginRight: '10px' }}
          >
            Upload File
          </Button>
          <Button
            variant={inputMethod === 'manual' ? 'primary' : undefined}
            onClick={() => setInputMethod('manual')}
          >
            Manual Input
          </Button>
        </div>

        {inputMethod === 'file' ? (
          <div>
            <p style={{ marginBottom: '20px' }}>
              Upload a JSON file containing the transition matrix and state names.
            </p>
            <FileUploader onFileSelect={handleFileUpload} accept=".json" />
          </div>
        ) : (
          <form onSubmit={handleSubmit(handleManualSubmit)}>
            <FormGroup>
              <Label htmlFor="manualInput">Enter Markov Chain Data (JSON format)</Label>
              <TextArea
                id="manualInput"
                {...register('manualInput', {
                  required: 'Input is required',
                  validate: value => {
                    try {
                      JSON.parse(value);
                      return true;
                    } catch (e) {
                      return 'Invalid JSON format';
                    }
                  }
                })}
                placeholder={`{
  "transition_matrix": [
    [0.3, 0.3, 0.2, 0.1, 0.1],
    [0.2, 0.2, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.4, 0.2, 0.1],
    [0.1, 0.1, 0.2, 0.5, 0.1],
    [0.2, 0.1, 0.1, 0.1, 0.5]
  ],
  "states": ["Twitter", "Facebook", "Instagram", "WhatsApp", "TikTok"]
}`}
              />
              {errors.manualInput && (
                <ErrorMessage>{errors.manualInput.message}</ErrorMessage>
              )}
            </FormGroup>
            <Button type="submit" variant="primary">Analyze</Button>
          </form>
        )}

        {loading && <Spinner />}

        {error && (
          <ErrorMessage style={{ marginTop: '20px' }}>{error}</ErrorMessage>
        )}
      </Card>

      {results && (
        <MarkovResults results={results} />
      )}
    </div>
  );
};

export default MarkovPage;
```

### 6.2 Components

#### Layout Component

```tsx
import React from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { theme, GlobalStyle } from '../../styles/theme';
import Navbar from './Navbar';

const LayoutContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.xl};
  width: 100%;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.spacing.lg};
  }
`;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle theme={theme} />
      <LayoutContainer>
        <Navbar />
        <MainContent>{children}</MainContent>
      </LayoutContainer>
    </ThemeProvider>
  );
};

export default Layout;
```

#### Navbar Component

```tsx
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import Link from 'next/link';
import { useRouter } from 'next/router';

const NavbarContainer = styled.header`
  background-color: ${({ theme }) => theme.colors.surface};
  padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.xl}`};
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: ${({ theme }) => theme.shadows.medium};
`;

const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: ${({ theme }) => theme.typography.fontWeights.bold};
  color: ${({ theme }) => theme.colors.primary};
  letter-spacing: 1px;
`;

const NavLinks = styled.nav`
  display: flex;
  gap: ${({ theme }) => theme.spacing.lg};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const NavLink = styled.a<{ active?: boolean }>`
  color: ${({ theme, active }) => active ? theme.colors.primary : theme.colors.textSecondary};
  text-decoration: none;
  font-weight: ${({ theme, active }) => active ? theme.typography.fontWeights.semibold : theme.typography.fontWeights.regular};
  padding: ${({ theme }) => `${theme.spacing.xs} ${theme.spacing.sm}`};
  border-bottom: 2px solid ${({ theme, active }) => active ? theme.colors.primary : 'transparent'};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.primary};
    border-bottom-color: ${({ theme }) => theme.colors.primary}80;
  }
`;

const MobileMenuButton = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.textPrimary};
  font-size: 1.5rem;
  cursor: pointer;
  display: none;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: block;
  }
`;

const MobileMenu = styled.div<{ isOpen: boolean }>`
  position: fixed;
  top: 60px;
  left: 0;
  right: 0;
  background-color: ${({ theme }) => theme.colors.surface};
  padding: ${({ theme }) => theme.spacing.md};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
  transform: translateY(${({ isOpen }) => isOpen ? '0' : '-100%'});
  opacity: ${({ isOpen }) => isOpen ? '1' : '0'};
  transition: all ${({ theme }) => theme.transitions.medium};
  box-shadow: ${({ theme }) => theme.shadows.medium};
  z-index: 99;

  @media (min-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const MobileNavLink = styled(NavLink)`
  padding: ${({ theme }) => theme.spacing.md};
  border-bottom: 1px solid ${({ theme }) => theme.colors.surfaceLight};

  &:last-child {
    border-bottom: none;
  }
`;

const Navbar: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const router = useRouter();

  const isActive = (path: string) => router.pathname === path;

  // Close mobile menu when route changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [router.pathname]);

  return (
    <>
      <NavbarContainer>
        <Logo>TehqeeqCast</Logo>

        <NavLinks>
          <Link href="/" passHref legacyBehavior>
            <NavLink active={isActive('/')}>Home</NavLink>
          </Link>
          <Link href="/markov" passHref legacyBehavior>
            <NavLink active={isActive('/markov')}>Markov Chain</NavLink>
          </Link>
          <Link href="/hmm" passHref legacyBehavior>
            <NavLink active={isActive('/hmm')}>Hidden Markov Model</NavLink>
          </Link>
          <Link href="/queue" passHref legacyBehavior>
            <NavLink active={isActive('/queue')}>M/M/1 Queue</NavLink>
          </Link>
        </NavLinks>

        <MobileMenuButton onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
          {mobileMenuOpen ? '✕' : '☰'}
        </MobileMenuButton>
      </NavbarContainer>

      <MobileMenu isOpen={mobileMenuOpen}>
        <Link href="/" passHref legacyBehavior>
          <MobileNavLink active={isActive('/')}>Home</MobileNavLink>
        </Link>
        <Link href="/markov" passHref legacyBehavior>
          <MobileNavLink active={isActive('/markov')}>Markov Chain</MobileNavLink>
        </Link>
        <Link href="/hmm" passHref legacyBehavior>
          <MobileNavLink active={isActive('/hmm')}>Hidden Markov Model</MobileNavLink>
        </Link>
        <Link href="/queue" passHref legacyBehavior>
          <MobileNavLink active={isActive('/queue')}>M/M/1 Queue</MobileNavLink>
        </Link>
      </MobileMenu>
    </>
  );
};

export default Navbar;
```

## 7. Conclusion

TehqeeqCast successfully demonstrates the application of stochastic processes to model and predict fake news propagation across social media platforms. By implementing Markov Chains, Hidden Markov Models, and M/M/1 Queues, the system provides valuable insights into how misinformation spreads, how to infer content credibility from observable metrics, and how to optimize content moderation resources. The web-based interface makes these complex mathematical models accessible to users without specialized knowledge, enabling better understanding and decision-making in combating misinformation. The project highlights the power of combining mathematical modeling with modern web technologies to address real-world challenges in the digital information ecosystem. Future work could extend these models to incorporate network effects, user influence factors, and more complex queueing systems for priority-based content moderation.

*********************END********************
