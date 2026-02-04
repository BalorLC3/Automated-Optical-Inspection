import { useState } from 'react';
import StatusCard from './components/StatusCard';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setProcessedImage(null);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setProcessedImage(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:8080/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Server error');

      const data = await response.json();
      setResult(data);
      
      // Set processed image if available
      if (data.data?.processedImage) {
        setProcessedImage(data.data.processedImage);
      }
    } catch (err) {
      console.error(err);
      setError('Failed to connect to server. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* HEADER */}
      <header className="w-full max-w-5xl text-center mb-10">
        <h1>Steel Defect Detection</h1>
        <p>Automated Optical Inspection (AOI)</p>
      </header>

      {/* MAIN GRID */}
      <main className="w-full max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-8">
        
        {/* LEFT COLUMN: Controls */}
        <div className="md:col-span-1 flex flex-col items-center space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-md w-full max-w-sm">
            <h2>Upload Image</h2>
            
            <input 
              type="file" 
              accept="image/*"
              onChange={handleFileChange}
              className="mb-4"
            />
            
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              className="w-full"
            >
              {loading ? 'Processing...' : 'Analyze Image'}
            </button>
            
            {error && (
              <div className="alert-error">
                {error}
              </div>
            )}
          </div>

          {/* Results Summary */}
          {result && (
            <StatusCard 
              qcStatus={result.qcStatus} 
              message={result.message} 
            />
          )}
        </div>

        {/* RIGHT COLUMN: Image Display */}
        <div className="md:col-span-2">
          {!preview ? (
            <div className="text-gray-400 text-center">
              <p>No image selected</p>
            </div>
          ) : (
            <div className="canvas-container">
              {processedImage ? (
                // Show processed image with boxes
                <img 
                  src={processedImage} 
                  alt="Processed with detections" 
                  className="canvas-display"
                />
              ) : (
                // Show original image
                <img 
                  src={preview} 
                  alt="Original" 
                  className="canvas-display"
                />
              )}
            </div>
          )}
        </div>

      </main>
    </>
  );
}

export default App;