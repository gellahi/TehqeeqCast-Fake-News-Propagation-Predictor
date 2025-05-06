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
  Divider
} from '../ui/StyledComponents';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MarkovResultsProps {
  results: {
    steady_state: Record<string, number>;
    mean_recurrence_times: Record<string, number>;
    mean_first_passage_times: Record<string, Record<string, number>>;
    transition_diagram: string;
  };
}

const MarkovResults: React.FC<MarkovResultsProps> = ({ results }) => {
  // Prepare data for the steady state chart
  const steadyStateData = Object.entries(results.steady_state).map(([state, probability]) => ({
    state,
    probability: parseFloat(probability.toFixed(4))
  }));
  
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
        <SectionTitle>Mean Recurrence Times</SectionTitle>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>State</TableHeaderCell>
              <TableHeaderCell>Mean Recurrence Time</TableHeaderCell>
            </TableRow>
          </TableHead>
          <tbody>
            {Object.entries(results.mean_recurrence_times).map(([state, time]) => (
              <TableRow key={state}>
                <TableCell>{state}</TableCell>
                <TableCell>{time.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </tbody>
        </Table>
      </Card>
      
      <Card>
        <SectionTitle>Mean First Passage Times</SectionTitle>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>From</TableHeaderCell>
              <TableHeaderCell>To</TableHeaderCell>
              <TableHeaderCell>Mean Passage Time</TableHeaderCell>
            </TableRow>
          </TableHead>
          <tbody>
            {Object.entries(results.mean_first_passage_times).flatMap(([fromState, toStates]) =>
              Object.entries(toStates).map(([toState, time]) => (
                <TableRow key={`${fromState}-${toState}`}>
                  <TableCell>{fromState}</TableCell>
                  <TableCell>{toState}</TableCell>
                  <TableCell>{time.toFixed(4)}</TableCell>
                </TableRow>
              ))
            )}
          </tbody>
        </Table>
      </Card>
      
      <Card>
        <SectionTitle>Transition Diagram</SectionTitle>
        <ImageContainer>
          <img src={results.transition_diagram} alt="Transition Diagram" />
        </ImageContainer>
      </Card>
    </div>
  );
};

export default MarkovResults;
