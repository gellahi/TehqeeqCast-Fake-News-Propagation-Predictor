import React, { useRef, useState } from 'react';
import { FileUpload, Button } from '../ui/StyledComponents';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  label?: string;
}

const FileUploader: React.FC<FileUploaderProps> = ({ 
  onFileSelect, 
  accept = '.json,.csv', 
  label = 'Drag & drop a file here, or click to select' 
}) => {
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      setFileName(file.name);
      onFileSelect(file);
    }
  };
  
  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      setFileName(file.name);
      onFileSelect(file);
    }
  };
  
  return (
    <FileUpload 
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input 
        type="file" 
        ref={fileInputRef}
        onChange={handleFileChange}
        accept={accept}
        style={{ display: 'none' }}
      />
      
      {fileName ? (
        <div>
          <p>Selected file: {fileName}</p>
          <Button 
            variant="secondary" 
            onClick={(e) => {
              e.stopPropagation();
              setFileName(null);
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
              }
            }}
          >
            Clear
          </Button>
        </div>
      ) : (
        <div>
          <p>{label}</p>
          <Button variant="primary">Select File</Button>
        </div>
      )}
    </FileUpload>
  );
};

export default FileUploader;
