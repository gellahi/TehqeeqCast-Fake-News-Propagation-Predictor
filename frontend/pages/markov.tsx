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
              Upload a JSON file containing the transition matrix and states for your Markov chain.
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
