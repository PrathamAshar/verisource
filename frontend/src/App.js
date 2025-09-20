import React, { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Progress } from './components/ui/progress';
import { Separator } from './components/ui/separator';
import { Camera, Shield, CheckCircle, AlertTriangle, Upload, Hash, Clock, Award } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Camera Capture Component
const CameraCapture = ({ onImageCaptured, isCapturing, setIsCapturing }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState('');

  const startCamera = useCallback(async () => {
    try {
      setError('');
      const newStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'environment' 
        } 
      });
      setStream(newStream);
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
      }
    } catch (err) {
      setError('Failed to access camera: ' + err.message);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [stream]);

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    context.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64Data = reader.result;
        onImageCaptured(base64Data);
        setIsCapturing(false);
        stopCamera();
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.9);
  }, [onImageCaptured, setIsCapturing, stopCamera]);

  useEffect(() => {
    if (isCapturing && !stream) {
      startCamera();
    } else if (!isCapturing && stream) {
      stopCamera();
    }
  }, [isCapturing, stream, startCamera, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  if (error) {
    return (
      <Alert className="mb-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {stream && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
            <Button 
              onClick={captureImage}
              size="lg"
              className="bg-red-600 hover:bg-red-700 text-white rounded-full w-16 h-16 p-0"
            >
              <Camera className="h-8 w-8" />
            </Button>
          </div>
        )}
      </div>
      
      {!stream && (
        <Button onClick={startCamera} className="w-full">
          <Camera className="mr-2 h-4 w-4" />
          Start Camera
        </Button>
      )}
    </div>
  );
};

// Trust Score Display Component
const TrustScoreDisplay = ({ trustScore, analysis }) => {
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 60) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (score >= 40) return 'text-orange-600 bg-orange-50 border-orange-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'HIGHLY_AUTHENTIC':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'LIKELY_AUTHENTIC':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'UNCERTAIN':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
    }
  };

  if (!analysis) return null;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Award className="h-5 w-5" />
          Trust Score Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Score */}
        <div className={`p-6 rounded-lg border-2 ${getScoreColor(trustScore)}`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {getVerdictIcon(analysis.overall_verdict)}
              <span className="font-semibold">
                {analysis.overall_verdict.replace('_', ' ')}
              </span>
            </div>
            <div className="text-2xl font-bold">{Math.round(trustScore)}/100</div>
          </div>
          <Progress value={trustScore} className="mb-2" />
          <p className="text-sm opacity-75">{analysis.explanation}</p>
        </div>

        {/* Detailed Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Blockchain Provenance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm">Score</span>
                <Badge variant="outline">{analysis.blockchain_provenance.score}/40</Badge>
              </div>
              <Progress value={(analysis.blockchain_provenance.score / 40) * 100} className="mb-2" />
              <p className="text-xs text-gray-600">
                {analysis.blockchain_provenance.verified ? 'Hash verified on blockchain' : 'Hash not found on blockchain'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">C2PA Integrity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm">Score</span>
                <Badge variant="outline">{analysis.c2pa_integrity.score}/25</Badge>
              </div>
              <Progress value={(analysis.c2pa_integrity.score / 25) * 100} className="mb-2" />
              <p className="text-xs text-gray-600">
                {analysis.c2pa_integrity.manifest_present ? 'Manifest present and valid' : 'No manifest found'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">AI Tamper Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm">Score</span>
                <Badge variant="outline">{Math.round(analysis.ai_tamper_analysis.score)}/25</Badge>
              </div>
              <Progress value={(analysis.ai_tamper_analysis.score / 25) * 100} className="mb-2" />
              <p className="text-xs text-gray-600">
                {Math.round(analysis.ai_tamper_analysis.tamper_probability)}% tamper probability
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Derivative Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm">Score</span>
                <Badge variant="outline">{analysis.derivative_analysis.score}/10</Badge>
              </div>
              <Progress value={(analysis.derivative_analysis.score / 10) * 100} className="mb-2" />
              <p className="text-xs text-gray-600">
                {analysis.derivative_analysis.analysis}
              </p>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
};

// Main App Components
const CaptureTab = () => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleImageCaptured = async (imageData) => {
    setCapturedImage(imageData);
    setIsProcessing(true);
    setError('');

    try {
      const response = await axios.post(`${API}/capture`, {
        image_data: imageData,
        metadata: {
          capture_method: 'web_camera',
          user_agent: navigator.userAgent,
          timestamp: new Date().toISOString()
        }
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setCapturedImage(null);
    setResult(null);
    setError('');
    setIsCapturing(false);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Capture & Authenticate
          </CardTitle>
          <CardDescription>
            Take a photo to create blockchain-verified provenance
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!capturedImage ? (
            <>
              {!isCapturing ? (
                <Button 
                  onClick={() => setIsCapturing(true)}
                  className="w-full mb-4"
                  size="lg"
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Start Camera Capture
                </Button>
              ) : (
                <CameraCapture 
                  onImageCaptured={handleImageCaptured}
                  isCapturing={isCapturing}
                  setIsCapturing={setIsCapturing}
                />
              )}
            </>
          ) : (
            <div className="space-y-4">
              <div className="relative bg-gray-100 rounded-lg overflow-hidden">
                <img 
                  src={capturedImage} 
                  alt="Captured" 
                  className="w-full h-auto max-h-96 object-contain"
                />
              </div>
              
              {isProcessing && (
                <div className="text-center space-y-2">
                  <div className="animate-pulse">Processing image...</div>
                  <Progress value={75} />
                </div>
              )}

              {error && (
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {result && (
                <Card className="bg-green-50 border-green-200">
                  <CardHeader>
                    <CardTitle className="text-green-800 flex items-center gap-2">
                      <CheckCircle className="h-5 w-5" />
                      Successfully Captured
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <strong>Image Hash:</strong>
                        <div className="font-mono text-xs bg-white p-2 rounded border mt-1">
                          {result.image_hash}
                        </div>
                      </div>
                      <div>
                        <strong>Receipt ID:</strong>
                        <div className="font-mono text-xs bg-white p-2 rounded border mt-1">
                          {result.receipt_id}
                        </div>
                      </div>
                    </div>
                    <div>
                      <strong>Blockchain Status:</strong>
                      <Badge variant="secondary" className="ml-2">
                        {result.blockchain_status}
                      </Badge>
                    </div>
                    <p className="text-sm text-green-700">{result.message}</p>
                  </CardContent>
                </Card>
              )}

              <Button onClick={reset} variant="outline" className="w-full">
                Capture Another Image
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const VerifyTab = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);
  const [error, setError] = useState('');

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setVerificationResult(null);
        setError('');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setVerificationResult(null);
        setError('');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const verifyImage = async () => {
    if (!selectedImage) return;

    setIsVerifying(true);
    setError('');

    try {
      const response = await axios.post(`${API}/verify`, {
        image_data: selectedImage
      });

      setVerificationResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to verify image');
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Verify Image Authenticity
          </CardTitle>
          <CardDescription>
            Upload an image to check its blockchain verification status
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!selectedImage ? (
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drop your image here, or click to select
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Supports JPG, PNG, and other common image formats
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
                id="image-upload"
              />
              <Button asChild>
                <label htmlFor="image-upload" className="cursor-pointer">
                  Select Image
                </label>
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative bg-gray-100 rounded-lg overflow-hidden">
                <img 
                  src={selectedImage} 
                  alt="Selected for verification" 
                  className="w-full h-auto max-h-96 object-contain"
                />
              </div>

              <div className="flex gap-2">
                <Button 
                  onClick={verifyImage} 
                  disabled={isVerifying}
                  className="flex-1"
                >
                  {isVerifying ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Verifying...
                    </>
                  ) : (
                    <>
                      <Shield className="mr-2 h-4 w-4" />
                      Verify Authenticity
                    </>
                  )}
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setSelectedImage(null);
                    setVerificationResult(null);
                    setError('');
                  }}
                >
                  Clear
                </Button>
              </div>

              {error && (
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {verificationResult && (
                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Hash className="h-5 w-5" />
                        Verification Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <strong>Image Hash:</strong>
                        <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1 break-all">
                          {verificationResult.image_hash}
                        </div>
                      </div>

                      <div>
                        <strong>Blockchain Status:</strong>
                        <Badge 
                          variant={verificationResult.blockchain_verification.verified ? "default" : "secondary"}
                          className="ml-2"
                        >
                          {verificationResult.blockchain_verification.verified ? 'VERIFIED' : 'NOT FOUND'}
                        </Badge>
                      </div>

                      {verificationResult.blockchain_verification.verified && (
                        <div className="bg-green-50 p-3 rounded border border-green-200">
                          <div className="text-sm space-y-1">
                            <div><strong>Batch ID:</strong> {verificationResult.blockchain_verification.batch_id}</div>
                            <div><strong>Transaction:</strong> 
                              <span className="font-mono text-xs ml-1">
                                {verificationResult.blockchain_verification.transaction_hash}
                              </span>
                            </div>
                            <div><strong>Block Number:</strong> {verificationResult.blockchain_verification.block_number}</div>
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Clock className="h-4 w-4" />
                        Verified at: {new Date(verificationResult.timestamp).toLocaleString()}
                      </div>
                    </CardContent>
                  </Card>

                  <TrustScoreDisplay 
                    trustScore={verificationResult.trust_score_analysis.trust_score}
                    analysis={verificationResult.trust_score_analysis}
                  />
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const StatsTab = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API}/stats`);
        setStats(response.data);
      } catch (err) {
        console.error('Failed to fetch stats:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="text-center">Loading statistics...</div>;
  }

  if (!stats) {
    return <div className="text-center text-gray-500">Failed to load statistics</div>;
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>System Statistics</CardTitle>
          <CardDescription>Real-time metrics from the VeriSource network</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-blue-600">
                  {stats.total_images_captured}
                </div>
                <p className="text-sm text-gray-600">Images Captured</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-green-600">
                  {stats.total_verified_images}
                </div>
                <p className="text-sm text-gray-600">Blockchain Verified</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-yellow-600">
                  {stats.pending_batch_size}
                </div>
                <p className="text-sm text-gray-600">Pending Batch</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-purple-600">
                  {stats.committed_batches}
                </div>
                <p className="text-sm text-gray-600">Committed Batches</p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            VeriSource
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Capture-Time Content Authenticity for the Web3 Era
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Blockchain-verified media provenance with AI-powered tamper detection
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <Tabs defaultValue="capture" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="capture" className="flex items-center gap-2">
                <Camera className="h-4 w-4" />
                Capture
              </TabsTrigger>
              <TabsTrigger value="verify" className="flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Verify
              </TabsTrigger>
              <TabsTrigger value="stats" className="flex items-center gap-2">
                <Award className="h-4 w-4" />
                Stats
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="capture" className="mt-6">
              <CaptureTab />
            </TabsContent>
            
            <TabsContent value="verify" className="mt-6">
              <VerifyTab />
            </TabsContent>
            
            <TabsContent value="stats" className="mt-6">
              <StatsTab />
            </TabsContent>
          </Tabs>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 pt-8 border-t border-gray-200">
          <p className="text-sm text-gray-500">
            Powered by blockchain technology, AI analysis, and C2PA standards
          </p>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;