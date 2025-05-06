# TehqeeqCast Sample Data

This folder contains sample data files for testing the TehqeeqCast application's stochastic models.

## Files

### 1. markov_chain.json

Contains data for the Markov Chain model that represents how fake news propagates across different social media platforms.

**Key components:**
- `transition_matrix`: Probabilities of content moving between platforms
- `states`: Names of the social media platforms
- `examples`: Sample content for testing

### 2. hidden_markov_model.json

Contains data for the Hidden Markov Model that represents the evolution of a post's credibility over time.

**Key components:**
- `hidden_states`: True credibility levels (True, Partially True, Fake)
- `observation_symbols`: Engagement metrics that can be observed
- `transition_matrix`: Probabilities of transitioning between hidden states
- `emission_matrix`: Probabilities of observing different engagement levels given a hidden state
- `initial_probabilities`: Starting distribution of hidden states
- `observations`: Sample observation sequence for testing

### 3. mm1_queue.json

Contains data for the M/M/1 Queue model that represents the content moderation pipeline.

**Key components:**
- `arrival_rate`: Rate at which posts are flagged for review (λ)
- `service_rate`: Rate at which moderators can process content (μ)
- `time_points`: Time points for plotting queue metrics
- `scenarios`: Different moderation scenarios with varying parameters

## Usage

These files can be uploaded through the TehqeeqCast web interface to test the different models:

1. Navigate to the appropriate model page (Markov Chain, HMM, or M/M/1 Queue)
2. Select "Upload File" option
3. Choose the corresponding JSON file from this folder
4. Click "Analyze" to process the data and view the results

## Extending the Data

Feel free to modify these files or create new ones to test different scenarios. Ensure that:

- All probability matrices have rows that sum to 1.0
- The dimensions of matrices match the number of states/symbols
- Arrival rate is less than service rate for stable M/M/1 queues
