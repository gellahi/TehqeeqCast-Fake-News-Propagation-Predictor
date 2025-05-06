import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { 
  PageTitle, 
  Card, 
  Button, 
  FormGroup, 
  Label, 
  TextArea,
  Spinner,
  ErrorMessage
} from '../components/ui/StyledComponents';
import FileUploader from '../components/common/FileUploader';
import HMMResults from '../components/results/HMMResults';
import { hmmApi } from '../services/api';

interface HMMFormData {
  manualInput: string;
}

const HMMPage: React.FC = () => {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inputMethod, setInputMethod] = useState<'file' | 'manual'>('file');
  
  const { register, handleSubmit, formState: { errors } } = useForm<HMMFormData>();
  
  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await hmmApi.upload(file);
      setResults(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred while processing the file');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleManualSubmit = async (data: HMMFormData) => {
    try {
      setLoading(true);
      setError(null);
      
      // Parse the manual input as JSON
      const parsedData = JSON.parse(data.manualInput);
      const results = await hmmApi.analyze(parsedData);
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
      <PageTitle>Hidden Markov Model Analysis</PageTitle>
      
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
              Upload a JSON file containing the HMM parameters and observation sequence.
            </p>
            <FileUploader onFileSelect={handleFileUpload} accept=".json" />
          </div>
        ) : (
          <form onSubmit={handleSubmit(handleManualSubmit)}>
            <FormGroup>
              <Label htmlFor="manualInput">Enter HMM Data (JSON format)</Label>
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
  "hidden_states": ["True", "Partially_True", "Fake"],
  "observation_symbols": ["Low_Engagement", "Medium_Engagement", "High_Engagement", "Viral"],
  "transition_matrix": [
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6]
  ],
  "emission_matrix": [
    [0.6, 0.3, 0.1, 0.0],
    [0.2, 0.4, 0.3, 0.1],
    [0.1, 0.2, 0.3, 0.4]
  ],
  "initial_probabilities": [0.5, 0.3, 0.2],
  "observations": [0, 1, 1, 2, 3, 3, 2, 1, 0]
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
        <HMMResults results={results} />
      )}
    </div>
  );
};

export default HMMPage;
