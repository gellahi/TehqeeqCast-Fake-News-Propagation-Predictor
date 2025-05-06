import React from 'react';
import {
  Card,
  SectionTitle,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableHeaderCell,
  ImageContainer,
  Badge,
  Divider
} from '../ui/StyledComponents';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface HMMResultsProps {
  results: {
    steady_state: Record<string, number>;
    observation_likelihood: number;
    most_likely_path: string[];
    state_probabilities: Record<string, number>[];
    path_diagram: string;
  };
}

const HMMResults: React.FC<HMMResultsProps> = ({ results }) => {
  // Prepare data for the steady state chart
  const steadyStateData = Object.entries(results.steady_state).map(([state, probability]) => ({
    state,
    probability: parseFloat(probability.toFixed(4))
  }));
  
  // Prepare data for the state probabilities over time
  const stateProbs = results.state_probabilities;
  const stateProbsData = stateProbs.map((probs, index) => {
    return {
      step: index,
      ...Object.entries(probs).reduce((acc, [state, prob]) => {
        acc[state] = parseFloat(prob.toFixed(4));
        return acc;
      }, {} as Record<string, number>)
    };
  });
  
  // Get all state names for the line chart
  const stateNames = Object.keys(results.steady_state);
  
  // Generate random colors for each state
  const getRandomColor = () => {
    const colors = ['#00ffaa', '#ff00aa', '#aa00ff', '#ffaa00', '#00aaff'];
    return colors[Math.floor(Math.random() * colors.length)];
  };
  
  const stateColors = stateNames.reduce((acc, state) => {
    acc[state] = getRandomColor();
    return acc;
  }, {} as Record<string, string>);
  
  return (
    <div>
      <Card>
        <SectionTitle>Steady State Probabilities</SectionTitle>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={steadyStateData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="state" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="probability" fill="#00ffaa" />
          </BarChart>
        </ResponsiveContainer>
        
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>State</TableHeaderCell>
              <TableHeaderCell>Probability</TableHeaderCell>
            </TableRow>
          </TableHead>
          <tbody>
            {Object.entries(results.steady_state).map(([state, probability]) => (
              <TableRow key={state}>
                <TableCell>{state}</TableCell>
                <TableCell>{probability.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </tbody>
        </Table>
      </Card>
      
      <Card>
        <SectionTitle>Observation Likelihood</SectionTitle>
        <p>The likelihood of the observed sequence: {results.observation_likelihood.toExponential(4)}</p>
      </Card>
      
      <Card>
        <SectionTitle>Most Likely Path (Viterbi)</SectionTitle>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
          {results.most_likely_path.map((state, index) => (
            <Badge key={index} variant={state === 'True' ? 'primary' : state === 'Fake' ? 'error' : 'secondary'}>
              {index}: {state}
            </Badge>
          ))}
        </div>
      </Card>
      
      <Card>
        <SectionTitle>State Probabilities Over Time</SectionTitle>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={stateProbsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis />
            <Tooltip />
            <Legend />
            {stateNames.map(state => (
              <Line 
                key={state}
                type="monotone"
                dataKey={state}
                stroke={stateColors[state]}
                activeDot={{ r: 8 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>
      
      <Card>
        <SectionTitle>Path Diagram</SectionTitle>
        <ImageContainer>
          <img src={results.path_diagram} alt="Path Diagram" />
        </ImageContainer>
      </Card>
    </div>
  );
};

export default HMMResults;
