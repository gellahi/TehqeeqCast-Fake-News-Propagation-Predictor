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

interface QueueResultsProps {
  results: {
    utilization: number;
    average_queue_length: number;
    average_system_length: number;
    average_wait_time: number;
    average_system_time: number;
    probability_idle: number;
    stability: boolean;
    queue_diagram: string;
  };
}

const QueueResults: React.FC<QueueResultsProps> = ({ results }) => {
  return (
    <div>
      <Card>
        <SectionTitle>Queue Stability</SectionTitle>
        <div style={{ marginBottom: '16px' }}>
          <Badge variant={results.stability ? 'primary' : 'error'}>
            {results.stability ? 'Stable' : 'Unstable'}
          </Badge>
        </div>
        
        <p>
          The queue is {results.stability ? 'stable' : 'unstable'}. 
          {!results.stability && ' The arrival rate exceeds the service rate, leading to an infinitely growing queue.'}
        </p>
      </Card>
      
      <Card>
        <SectionTitle>Queue Metrics</SectionTitle>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Metric</TableHeaderCell>
              <TableHeaderCell>Value</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
            </TableRow>
          </TableHead>
          <tbody>
            <TableRow>
              <TableCell>Server Utilization (ρ)</TableCell>
              <TableCell>{results.utilization.toFixed(4)}</TableCell>
              <TableCell>Fraction of time the server is busy</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Probability Idle (P₀)</TableCell>
              <TableCell>{results.probability_idle.toFixed(4)}</TableCell>
              <TableCell>Probability that the system is empty</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Average Queue Length (Lq)</TableCell>
              <TableCell>
                {Number.isFinite(results.average_queue_length) 
                  ? results.average_queue_length.toFixed(4) 
                  : '∞'}
              </TableCell>
              <TableCell>Average number of customers waiting in queue</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Average System Length (L)</TableCell>
              <TableCell>
                {Number.isFinite(results.average_system_length) 
                  ? results.average_system_length.toFixed(4) 
                  : '∞'}
              </TableCell>
              <TableCell>Average number of customers in the system</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Average Wait Time (Wq)</TableCell>
              <TableCell>
                {Number.isFinite(results.average_wait_time) 
                  ? results.average_wait_time.toFixed(4) 
                  : '∞'}
              </TableCell>
              <TableCell>Average time a customer spends waiting in queue</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Average System Time (W)</TableCell>
              <TableCell>
                {Number.isFinite(results.average_system_time) 
                  ? results.average_system_time.toFixed(4) 
                  : '∞'}
              </TableCell>
              <TableCell>Average time a customer spends in the system</TableCell>
            </TableRow>
          </tbody>
        </Table>
      </Card>
      
      <Card>
        <SectionTitle>Queue Diagrams</SectionTitle>
        <ImageContainer>
          <img src={results.queue_diagram} alt="Queue Diagram" />
        </ImageContainer>
      </Card>
    </div>
  );
};

export default QueueResults;
