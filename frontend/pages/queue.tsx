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
  ErrorMessage,
  Flex
} from '../components/ui/StyledComponents';
import FileUploader from '../components/common/FileUploader';
import QueueResults from '../components/results/QueueResults';
import { queueApi } from '../services/api';

interface QueueFormData {
  arrivalRate: number;
  serviceRate: number;
  manualInput: string;
}

const QueuePage: React.FC = () => {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inputMethod, setInputMethod] = useState<'file' | 'manual' | 'simple'>('simple');
  
  const { register, handleSubmit, formState: { errors } } = useForm<QueueFormData>({
    defaultValues: {
      arrivalRate: 5,
      serviceRate: 6
    }
  });
  
  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await queueApi.upload(file);
      setResults(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred while processing the file');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleManualSubmit = async (data: QueueFormData) => {
    try {
      setLoading(true);
      setError(null);
      
      let requestData;
      
      if (inputMethod === 'simple') {
        requestData = {
          arrival_rate: parseFloat(data.arrivalRate.toString()),
          service_rate: parseFloat(data.serviceRate.toString())
        };
      } else {
        // Parse the manual input as JSON
        requestData = JSON.parse(data.manualInput);
      }
      
      const results = await queueApi.analyze(requestData);
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
      <PageTitle>M/M/1 Queue Analysis</PageTitle>
      
      <Card>
        <div style={{ marginBottom: '20px' }}>
          <Button 
            variant={inputMethod === 'simple' ? 'primary' : undefined}
            onClick={() => setInputMethod('simple')}
            style={{ marginRight: '10px' }}
          >
            Simple Input
          </Button>
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
            Advanced Input
          </Button>
        </div>
        
        {inputMethod === 'file' ? (
          <div>
            <p style={{ marginBottom: '20px' }}>
              Upload a JSON file containing the arrival rate and service rate for your M/M/1 queue.
            </p>
            <FileUploader onFileSelect={handleFileUpload} accept=".json" />
          </div>
        ) : inputMethod === 'simple' ? (
          <form onSubmit={handleSubmit(handleManualSubmit)}>
            <Flex>
              <FormGroup style={{ flex: 1, marginRight: '16px' }}>
                <Label htmlFor="arrivalRate">Arrival Rate (λ)</Label>
                <Input 
                  id="arrivalRate"
                  type="number"
                  step="0.1"
                  {...register('arrivalRate', { 
                    required: 'Arrival rate is required',
                    min: { value: 0.1, message: 'Must be greater than 0' }
                  })}
                />
                {errors.arrivalRate && (
                  <ErrorMessage>{errors.arrivalRate.message}</ErrorMessage>
                )}
              </FormGroup>
              
              <FormGroup style={{ flex: 1 }}>
                <Label htmlFor="serviceRate">Service Rate (μ)</Label>
                <Input 
                  id="serviceRate"
                  type="number"
                  step="0.1"
                  {...register('serviceRate', { 
                    required: 'Service rate is required',
                    min: { value: 0.1, message: 'Must be greater than 0' }
                  })}
                />
                {errors.serviceRate && (
                  <ErrorMessage>{errors.serviceRate.message}</ErrorMessage>
                )}
              </FormGroup>
            </Flex>
            
            <Button type="submit" variant="primary">Analyze</Button>
          </form>
        ) : (
          <form onSubmit={handleSubmit(handleManualSubmit)}>
            <FormGroup>
              <Label htmlFor="manualInput">Enter Queue Data (JSON format)</Label>
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
  "arrival_rate": 5.0,
  "service_rate": 6.0,
  "time_points": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
        <QueueResults results={results} />
      )}
    </div>
  );
};

export default QueuePage;
