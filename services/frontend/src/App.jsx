import { useState } from 'react';
import ImageCanvas from './components/ImageCanvas';
import StatusCard from './components/StatusCard';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null); // Reset previous results
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      // Connect to Go Backend
      const response = await fetch('http://localhost:8080/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Server error');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to backend. Is Go running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 font-sans text-gray-800 flex flex-col items-center justify-center">
      <header className="max-w-5xl mx-auto mb-10 flex justify-end">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Steel Defect Detection</h1>
        <p className="text-gray-500">Automated Optical Inspection (AOI) Pipeline</p>
      </header>

      <main className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        
        {/* LEFT COLUMN: Controls */}
        <div className="md:col-span-1 space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h2 className="text-lg font-semibold mb-4">1. Upload Image</h2>
            <input 
              type="file" 
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100 mb-4"
            />
            
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-colors
                ${loading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700 shadow-lg'}`}
            >
              {loading ? 'Processing...' : 'Analyze Image'}
            </button>
            
            {error && (
              <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}
          </div>

          {/* Results Summary */}
          {result && (
             <StatusCard qcStatus={result.qc_status} message={result.message} />
          )}
        </div>

        {/* RIGHT COLUMN: Visualizer */}
        <div className="md:col-span-2 bg-white p-6 rounded-xl shadow-md flex flex-col items-center justify-center min-h-[400px]">
          {!preview && (
            <div className="text-gray-400 text-center">
              <p>No image selected</p>
            </div>
          )}

          {preview && (
            <ImageCanvas 
              imageSrc={preview} 
              detections={result?.data?.detections || []} 
            />
          )}
        </div>

      </main>
    </div>
  );
}

export default App;