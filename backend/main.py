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
