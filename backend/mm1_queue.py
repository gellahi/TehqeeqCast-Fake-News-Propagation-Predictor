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

# Helper functions for M/M/1 Queue calculations
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
    
    # 1. Plot state probabilities
    states = np.arange(0, 15)
    if metrics["stability"]:
        probabilities = [(1 - utilization) * (utilization ** n) for n in states]
    else:
        probabilities = [0 if n > 0 else 1 for n in states]
    
    axs[0, 0].bar(states, probabilities)
    axs[0, 0].set_title('State Probabilities P(n)')
    axs[0, 0].set_xlabel('Number of Customers (n)')
    axs[0, 0].set_ylabel('Probability')
    axs[0, 0].grid(True)
    
    # 2. Plot cumulative distribution function of waiting time
    if metrics["stability"]:
        wait_times = np.linspace(0, 5 * metrics["average_wait_time"], 100)
        cdf_values = 1 - np.exp(-service_rate * (1 - utilization) * wait_times)
    else:
        wait_times = np.linspace(0, 10, 100)
        cdf_values = np.zeros_like(wait_times)
    
    axs[0, 1].plot(wait_times, cdf_values)
    axs[0, 1].set_title('CDF of Waiting Time')
    axs[0, 1].set_xlabel('Waiting Time')
    axs[0, 1].set_ylabel('Probability (Wq ≤ t)')
    axs[0, 1].grid(True)
    
    # 3. Plot average queue length over time (transient behavior)
    if metrics["stability"]:
        # Simple approximation of transient behavior
        steady_state = metrics["average_queue_length"]
        transient = [steady_state * (1 - np.exp(-service_rate * (1 - utilization) * t)) for t in time_points]
    else:
        # Unstable queue grows linearly
        transient = [(arrival_rate - service_rate) * t for t in time_points]
    
    axs[1, 0].plot(time_points, transient)
    axs[1, 0].set_title('Average Queue Length Over Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Average Queue Length')
    axs[1, 0].grid(True)
    
    # 4. Plot server utilization vs arrival rate
    lambda_range = np.linspace(0, service_rate * 0.99, 100)
    utilizations = lambda_range / service_rate
    
    axs[1, 1].plot(lambda_range, utilizations)
    axs[1, 1].axvline(x=arrival_rate, color='r', linestyle='--', label=f'λ = {arrival_rate}')
    axs[1, 1].axhline(y=utilization, color='g', linestyle='--', label=f'ρ = {utilization:.2f}')
    axs[1, 1].set_title('Server Utilization vs Arrival Rate')
    axs[1, 1].set_xlabel('Arrival Rate (λ)')
    axs[1, 1].set_ylabel('Server Utilization (ρ)')
    axs[1, 1].set_xlim(0, service_rate)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Add overall title
    plt.suptitle(f'M/M/1 Queue Analysis (λ={arrival_rate}, μ={service_rate})', fontsize=16)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.savefig(buf, format='png')
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

@MM1QueueRouter.post("/upload")
async def upload_queue_data(file: UploadFile = File(...)):
    """Upload a JSON file with M/M/1 queue data."""
    try:
        content = await file.read()
        data = json.loads(content)
        
        # Validate the uploaded data
        if "arrival_rate" not in data or "service_rate" not in data:
            raise HTTPException(status_code=400, 
                               detail="Invalid data format. Must include 'arrival_rate' and 'service_rate'")
        
        # Process the data
        input_data = MM1QueueInput(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            time_points=data.get("time_points")
        )
        
        return await analyze_mm1_queue(input_data)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
