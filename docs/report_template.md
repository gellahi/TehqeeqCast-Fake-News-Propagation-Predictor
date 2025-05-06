# TehqeeqCast: Fake-News Propagation Predictor

## 1. Introduction

In the era of social media, the rapid spread of misinformation poses significant challenges to society. TehqeeqCast is a web application designed to model and predict the propagation of fake news across social media platforms using stochastic processes. This report outlines the mathematical models implemented, their applications to the fake news problem, and the results obtained from our analysis.

### 1.1 Problem Statement

The spread of misinformation on social media platforms has become a critical issue affecting public discourse, elections, and public health. Understanding how fake news propagates, how to identify it, and how to efficiently moderate content are essential challenges that require mathematical modeling and computational approaches.

### 1.2 Project Objectives

TehqeeqCast aims to:
- Model the transition of fake news across different social media platforms
- Infer the hidden credibility state of posts based on observable engagement metrics
- Forecast moderation backlogs in fact-checking pipelines
- Provide an interactive web interface for analyzing fake news propagation

## 2. Real-World Problem Context

### 2.1 Fake News Propagation Patterns

Misinformation typically follows specific patterns as it moves across platforms. For example, a false claim might originate on Twitter, gain traction on Facebook, spread through private WhatsApp groups, and eventually reach viral status on TikTok. Each platform has different characteristics that affect how content spreads and is consumed.

### 2.2 Content Credibility Evolution

The perceived credibility of content often changes over time. Initially, a fake news post might appear credible, but as fact-checkers investigate and debunk claims, its credibility decreases. However, this credibility state is not directly observable—we can only see engagement metrics like likes, shares, and comments.

### 2.3 Moderation Challenges

Content moderation teams face significant backlogs when reviewing flagged posts. Understanding queue dynamics helps platforms allocate resources efficiently to minimize the time that potentially harmful content remains visible.

## 3. Mathematical Models

### 3.1 Markov Chain for Platform Transitions

We model the movement of fake news across platforms as a Markov process, where:
- States represent different social media platforms (Twitter, Facebook, Instagram, WhatsApp, TikTok)
- Transition probabilities represent the likelihood of content moving from one platform to another
- Steady-state probabilities indicate the long-term distribution of fake news across platforms
- Mean first passage times reveal how quickly misinformation reaches specific platforms

#### 3.1.1 Mathematical Formulation

For a set of platforms $S = \{s_1, s_2, ..., s_n\}$, we define a transition matrix $P$ where $P_{ij}$ represents the probability of content moving from platform $i$ to platform $j$. The steady-state distribution $\pi$ satisfies:

$$\pi P = \pi \quad \text{and} \quad \sum_{i=1}^{n} \pi_i = 1$$

### 3.2 Hidden Markov Model for Credibility Evolution

We use an HMM to model the hidden credibility state of posts:
- Hidden states represent true credibility levels (True, Partially True, Fake)
- Observations are engagement metrics (Low, Medium, High, Viral engagement)
- Transition probabilities model how credibility evolves over time
- Emission probabilities link hidden credibility to observable engagement

#### 3.2.1 Mathematical Formulation

Our HMM is defined by:
- Hidden states $Q = \{q_1, q_2, ..., q_N\}$ (credibility levels)
- Observation symbols $O = \{o_1, o_2, ..., o_M\}$ (engagement levels)
- Transition probabilities $A = \{a_{ij}\}$ where $a_{ij} = P(q_j \text{ at } t+1 | q_i \text{ at } t)$
- Emission probabilities $B = \{b_i(k)\}$ where $b_i(k) = P(o_k | q_i)$
- Initial state distribution $\pi = \{\pi_i\}$

The Viterbi algorithm finds the most likely sequence of credibility states given the observed engagement metrics.

### 3.3 M/M/1 Queue for Moderation Pipeline

We model the content moderation process as an M/M/1 queue:
- Arrival rate λ represents the rate at which posts are flagged for review
- Service rate μ represents the rate at which moderators process content
- Queue metrics help forecast backlogs and resource requirements

#### 3.3.1 Mathematical Formulation

For an M/M/1 queue with arrival rate λ and service rate μ, key metrics include:
- Server utilization: $\rho = \lambda/\mu$
- Average queue length: $L_q = \rho^2/(1-\rho)$
- Average system length: $L = \rho/(1-\rho)$
- Average waiting time: $W_q = L_q/\lambda$
- Average system time: $W = L/\lambda$

## 4. Implementation and Results

### 4.1 Markov Chain Analysis

[Include sample results from the Markov chain analysis, such as:
- Transition diagram visualization
- Steady-state distribution across platforms
- Mean recurrence and first passage times
- Interpretation of results in the context of fake news]

### 4.2 Hidden Markov Model Analysis

[Include sample results from the HMM analysis, such as:
- Most likely credibility path for sample posts
- Observation likelihood
- State probability evolution over time
- Interpretation of how credibility evolves based on engagement]

### 4.3 M/M/1 Queue Analysis

[Include sample results from the queue analysis, such as:
- Queue stability analysis
- Average queue length and waiting time
- Server utilization
- Recommendations for moderation team sizing]

## 5. Web Application Implementation

### 5.1 Backend Architecture

The backend is implemented using FastAPI, with separate modules for each mathematical model:
- `markov_chain.py`: Implements Markov chain analysis
- `hidden_markov.py`: Implements HMM algorithms
- `mm1_queue.py`: Implements M/M/1 queue calculations

### 5.2 Frontend Design

The frontend is built with Next.js and TypeScript, featuring:
- Interactive forms for data input
- Visualizations using Recharts and base64-encoded images
- Dark-glossy theme with neon accents for readability

## 6. Conclusion and Future Work

### 6.1 Summary of Findings

[Summarize the key insights gained from the stochastic modeling of fake news propagation]

### 6.2 Limitations

[Discuss limitations of the current models and implementation]

### 6.3 Future Enhancements

Potential future enhancements include:
- Incorporating network effects and user influence in propagation models
- Implementing more complex queueing systems (e.g., priority queues for high-risk content)
- Adding machine learning components for automatic credibility assessment
- Extending the models to include temporal dynamics and external events

## 7. References

[List relevant academic papers, books, and online resources]
