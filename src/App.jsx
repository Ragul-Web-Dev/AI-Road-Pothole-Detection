import React, { useState, useEffect, useRef } from 'react';
import {
  SAMPLES,
  RESEARCH_PROGRESS_ITEMS,
  TIMELINE_DATA,
  METHODOLOGY_STEPS,
} from './data/mockData';
import RoadCanvas from './components/RoadCanvas';
import LiveCameraPanel from './components/LiveCameraPanel';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Circle,
  Clock,
  Database,
  Download,
  Eye,
  FileCheck,
  Gauge,
  HardDrive,
  ImageIcon,
  Info,
  Layers,
  Moon,
  Navigation,
  RefreshCw,
  RotateCcw,
  Route,
  Server,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Sun,
  Sunset,
  CloudRain,
  Target,
  UploadCloud,
  Volume2,
  XCircle,
  Zap,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

export default function App() {
  const [activeTab, setActiveTab] = useState('camera');

  const [legalSpeedLimit, setLegalSpeedLimit] = useState(50);

  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [condition, setCondition] = useState('Not specified');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ online: false, checking: true, info: null });
  const [viewMode, setViewMode] = useState('sideBySide');
  const [presetSamples, setPresetSamples] = useState([]);

  const fileInputRef = useRef(null);

  const [selectedSampleIndex, setSelectedSampleIndex] = useState(0);
  const [uploadedSampleImage, setUploadedSampleImage] = useState(null);
  const reviewFileRef = useRef(null);

  const currentSample = SAMPLES[selectedSampleIndex];

  const checkBackendHealth = async () => {
    setBackendStatus((prev) => ({ ...prev, checking: true }));
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        setBackendStatus({ online: true, checking: false, info: data });
        fetchPresetSamples();
      } else {
        setBackendStatus({ online: false, checking: false, info: null });
      }
    } catch {
      setBackendStatus({ online: false, checking: false, info: null });
    }
  };

  const fetchPresetSamples = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/samples`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        if (data.samples && data.samples.length > 0) {
          setPresetSamples(data.samples);
        }
      }
    } catch {
    }
  };

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const handleFileSelect = (file) => {
    if (!file) return;

    const validExtensions = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const fileName = file.name ? file.name.toLowerCase() : '';
    const hasValidExt =
      fileName.endsWith('.jpg') ||
      fileName.endsWith('.jpeg') ||
      fileName.endsWith('.png') ||
      fileName.endsWith('.webp');

    if (!validExtensions.includes(file.type) && !hasValidExt) {
      setErrorMessage(
        'Invalid file format. Please upload a valid JPG, JPEG, PNG, or WEBP road image.'
      );
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 20MB limit. Please select a smaller image.');
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    setDetectionResult(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleSelectPresetSample = (sample) => {
    setErrorMessage(null);
    setDetectionResult(null);
    setImagePreview(sample.data_url);
    setSelectedFile({ name: sample.filename });
  };

  const handleRunInference = async () => {
    if (!imagePreview) {
      setErrorMessage('Please upload or select a road image before running prediction.');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const payload = {
        image: imagePreview,
        conf: confidenceThreshold,
        legal_speed_limit: legalSpeedLimit,
        filename: selectedFile?.name || 'road_input_image.jpg',
        condition: condition,
      };

      const response = await fetch(`${API_BASE_URL}/api/detect/image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.error || `Server returned error status ${response.status}`);
      }

      const data = await response.json();
      setDetectionResult(data);
      setBackendStatus((prev) => ({ ...prev, online: true }));
    } catch (err) {
      console.error('Inference execution error:', err);
      setErrorMessage(
        `Backend inference server connection failed (${err.message || 'Server Offline'}). Please ensure the FastAPI server is running with 'python pothole-research/scripts/server.py'.`
      );
      setBackendStatus((prev) => ({ ...prev, online: false }));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setDetectionResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDownloadAnnotatedImage = () => {
    if (!detectionResult?.annotated_image) return;
    const link = document.createElement('a');
    link.href = detectionResult.annotated_image;
    link.download = `pothole_annotated_${detectionResult.original_filename || 'result'}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const conditionIcons = {
    Day: <Sun className="w-3.5 h-3.5 text-amber-500" />,
    Evening: <Sunset className="w-3.5 h-3.5 text-orange-500" />,
    Night: <Moon className="w-3.5 h-3.5 text-indigo-400" />,
    Rain: <CloudRain className="w-3.5 h-3.5 text-blue-500" />,
    'Not specified': <Circle className="w-3.5 h-3.5 text-slate-400" />,
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-50 text-blue-700 border border-blue-200">
                <FileCheck className="w-3.5 h-3.5" />
                AI-Based Road Hazard Early Detection System
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <Volume2 className="w-3 h-3 text-emerald-600" />
                Segment 4: Live Audio & Camera Active
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-black text-slate-900 mt-1 tracking-tight">
              Road Hazard Early Detection & Adaptive Safe-Speed Warning System
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 font-medium">
              Deep Learning Vision Pipeline • Ultralytics YOLOv8n Research Benchmark
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div
              onClick={checkBackendHealth}
              title="Click to re-check FastAPI backend health"
              className={`cursor-pointer inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono font-semibold border transition ${
                backendStatus.online
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100'
                  : 'bg-red-50 text-red-800 border-red-300 hover:bg-red-100'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus.online ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
                }`}
              ></span>
              <span>
                {backendStatus.checking
                  ? 'Checking API...'
                  : backendStatus.online
                  ? 'API Online (FastAPI)'
                  : 'API Offline'}
              </span>
              <RefreshCw className={`w-3 h-3 ml-1 ${backendStatus.checking ? 'animate-spin' : ''}`} />
            </div>

            <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
              <button
                onClick={() => setActiveTab('camera')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold transition ${
                  activeTab === 'camera'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Camera className="w-3.5 h-3.5" />
                Live Camera & Audio
              </button>

              <button
                onClick={() => setActiveTab('segment3')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold transition ${
                  activeTab === 'segment3'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <UploadCloud className="w-3.5 h-3.5" />
                Image Upload
              </button>

              <button
                onClick={() => setActiveTab('review')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold transition ${
                  activeTab === 'review'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                Phase 1 & 2 Prototype
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 flex-1 w-full">

        {activeTab === 'camera' && (
          <LiveCameraPanel
            backendOnline={backendStatus.online}
            legalSpeedLimit={legalSpeedLimit}
            setLegalSpeedLimit={setLegalSpeedLimit}
          />
        )}

        {activeTab === 'segment3' && (
          <div className="space-y-6">

            {!backendStatus.online && (
              <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 text-xs text-amber-900 flex items-start gap-3 shadow-sm">
                <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="font-bold text-amber-950">FastAPI ML Backend Server Disconnected</div>
                  <div className="mt-0.5 text-amber-800 leading-relaxed">
                    To perform genuine YOLOv8n pothole detection inference, start the Python FastAPI service in your terminal:
                  </div>
                  <div className="mt-2 text-[12px] font-mono bg-amber-100/90 p-2.5 rounded-md border border-amber-300 text-slate-900 select-all">
                    python pothole-research/scripts/server.py --port 5000
                  </div>
                </div>
                <button
                  onClick={checkBackendHealth}
                  className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold transition shrink-0"
                >
                  Retry Connection
                </button>
              </div>
            )}

            {errorMessage && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-xs text-red-800 flex items-start gap-3 shadow-sm">
                <XCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="font-bold text-red-900">Inference Error</div>
                  <div className="mt-0.5">{errorMessage}</div>
                </div>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="text-red-500 hover:text-red-700 font-bold"
                >
                  ✕
                </button>
              </div>
            )}

            {backendStatus.info && (
              <div className="bg-blue-50/70 border border-blue-200 rounded-xl p-3.5 text-xs text-blue-900 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-blue-600 shrink-0" />
                  <span className="font-semibold">Loaded Model:</span>
                  <span className="font-mono bg-white px-2 py-0.5 rounded border border-blue-200 text-blue-800 font-bold">
                    YOLOv8n (runs/detect/pothole_training/weights/best.pt)
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[11px] font-mono text-slate-600">
                  <span>Task: Object Detection</span>
                  <span>Classes: {backendStatus.info.model_classes ? Object.values(backendStatus.info.model_classes).join(', ') : 'pothole'}</span>
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
                <div>
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <UploadCloud className="w-5 h-5 text-blue-600" />
                    Image Upload & Prediction Parameters
                  </h2>
                  <p className="text-xs text-slate-500">
                    Upload a road surface image or pick a test benchmark sample to execute genuine YOLOv8n hazard detection.
                  </p>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs bg-slate-50 px-2.5 py-1 rounded border border-slate-200 text-slate-600">
                  <Server className="w-3.5 h-3.5 text-blue-600" />
                  POST /api/detect/image
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                <div className="lg:col-span-7 space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold text-slate-700 uppercase tracking-wide">
                      Step 1: Upload Road Image
                    </label>
                    <span className="text-[11px] text-slate-400 font-mono">JPG, PNG, WEBP (Max 20MB)</span>
                  </div>

                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={(e) => handleFileSelect(e.target.files?.[0])}
                    accept=".jpg,.jpeg,.png,.webp"
                    className="hidden"
                  />

                  {!imagePreview ? (
                    <div
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50/70 hover:bg-blue-50/40 rounded-xl p-8 text-center cursor-pointer transition flex flex-col items-center justify-center min-h-[260px] group"
                    >
                      <div className="w-14 h-14 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mb-3 group-hover:scale-105 transition shadow-sm">
                        <UploadCloud className="w-7 h-7" />
                      </div>
                      <div className="text-sm font-bold text-slate-800">
                        Upload Road Surface Image
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        Drag & drop image here, or <span className="text-blue-600 font-semibold underline">Browse Files</span>
                      </div>
                      <div className="text-[11px] text-slate-400 mt-3 font-mono">
                        Valid formats: JPG, JPEG, PNG, WEBP
                      </div>
                    </div>
                  ) : (
                    <div className="relative rounded-xl overflow-hidden border border-slate-300 bg-slate-900 group shadow-inner">
                      <img
                        src={imagePreview}
                        alt="Input road preview"
                        className="w-full max-h-[340px] object-contain mx-auto"
                      />
                      <div className="absolute bottom-2 left-2 bg-slate-900/80 text-white font-mono text-[11px] px-2.5 py-1 rounded backdrop-blur border border-slate-700">
                        {selectedFile?.name || 'Selected Image'}
                      </div>
                      <button
                        onClick={handleReset}
                        className="absolute top-2 right-2 px-2.5 py-1 rounded bg-red-600/90 hover:bg-red-700 text-white text-xs font-semibold shadow backdrop-blur transition"
                      >
                        Change Image
                      </button>
                    </div>
                  )}

                  {presetSamples.length > 0 && (
                    <div className="pt-2">
                      <div className="text-[11px] font-bold text-slate-600 uppercase mb-1.5">
                        Or select benchmark test dataset sample:
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {presetSamples.map((s) => (
                          <button
                            key={s.id}
                            type="button"
                            onClick={() => handleSelectPresetSample(s)}
                            className={`px-2.5 py-1 rounded text-xs font-semibold border transition ${
                              selectedFile?.name === s.filename
                                ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                                : s.category === 'normal'
                                ? 'bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100'
                                : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                            }`}
                          >
                            {s.category === 'normal' ? '🌱 ' : '⚠️ '}
                            {s.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
                  <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-bold text-slate-800 uppercase flex items-center gap-1.5">
                        <Gauge className="w-4 h-4 text-blue-600" />
                        Posted Legal Speed Limit
                      </label>
                      <span className="text-xs font-mono font-extrabold text-blue-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                        {legalSpeedLimit} km/h
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min="20"
                        max="120"
                        step="5"
                        value={legalSpeedLimit}
                        onChange={(e) => setLegalSpeedLimit(parseInt(e.target.value, 10))}
                        className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                      />
                    </div>
                    <div className="flex items-center justify-between gap-1 pt-1">
                      {[30, 40, 50, 60, 80, 100].map((spd) => (
                        <button
                          key={spd}
                          type="button"
                          onClick={() => setLegalSpeedLimit(spd)}
                          className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border transition ${
                            legalSpeedLimit === spd
                              ? 'bg-blue-600 text-white border-blue-600'
                              : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-100'
                          }`}
                        >
                          {spd}
                        </button>
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-500">
                      * Legal speed limit is posted by road authority and remains distinct from advisory warnings.
                    </p>
                  </div>

                  <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-bold text-slate-800 uppercase flex items-center gap-1.5">
                        <Sliders className="w-4 h-4 text-blue-600" />
                        Confidence Threshold
                      </label>
                      <span className="text-xs font-mono font-extrabold text-blue-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                        {confidenceThreshold.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                      <span>0.10 (High Recall)</span>
                      <span>0.25 (Default)</span>
                      <span>0.90 (High Precision)</span>
                    </div>
                  </div>

                  <div>
                    <label className="text-xs font-bold text-slate-700 uppercase mb-1.5 block">
                      Environmental Condition (Metadata)
                    </label>
                    <div className="grid grid-cols-3 sm:grid-cols-5 gap-1.5">
                      {['Day', 'Evening', 'Night', 'Rain', 'Not specified'].map((cond) => (
                        <button
                          key={cond}
                          type="button"
                          onClick={() => setCondition(cond)}
                          className={`flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg text-xs font-bold border transition ${
                            condition === cond
                              ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                              : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                          }`}
                        >
                          {conditionIcons[cond]}
                          <span className="truncate">{cond}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="pt-2">
                    <button
                      onClick={handleRunInference}
                      disabled={isProcessing || !imagePreview}
                      className={`w-full py-3.5 px-4 rounded-xl text-xs font-black uppercase tracking-wider flex items-center justify-center gap-2 transition shadow-md ${
                        isProcessing || !imagePreview
                          ? 'bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300'
                          : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-500/25 active:scale-[0.99]'
                      }`}
                    >
                      {isProcessing ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Running YOLOv8n Inference...</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4" />
                          <span>Predict Potholes (Run Model)</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {detectionResult && (
              <div className="space-y-6">

                <div
                  className={`rounded-xl border p-5 shadow-sm transition ${
                    detectionResult.potholes_detected > 0
                      ? 'bg-red-50/90 border-red-300 text-red-950'
                      : 'bg-emerald-50/90 border-emerald-300 text-emerald-950'
                  }`}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-start gap-3.5">
                      <div
                        className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 shadow-sm ${
                          detectionResult.potholes_detected > 0
                            ? 'bg-red-600 text-white'
                            : 'bg-emerald-600 text-white'
                        }`}
                      >
                        {detectionResult.potholes_detected > 0 ? (
                          <ShieldAlert className="w-7 h-7 animate-bounce" />
                        ) : (
                          <ShieldCheck className="w-7 h-7" />
                        )}
                      </div>

                      <div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-mono font-black uppercase tracking-wider ${
                              detectionResult.potholes_detected > 0
                                ? 'bg-red-200 text-red-900 border border-red-300'
                                : 'bg-emerald-200 text-emerald-900 border border-emerald-300'
                            }`}
                          >
                            {detectionResult.hazard_advisory?.level || 'ADVISORY ALERT'}
                          </span>
                          <span className="text-xs text-slate-500 font-mono">
                            {detectionResult.timestamp}
                          </span>
                        </div>

                        <h3 className="text-lg sm:text-xl font-black mt-1 tracking-tight">
                          {detectionResult.hazard_advisory?.alert_title ||
                            (detectionResult.potholes_detected > 0
                              ? 'POTHOLE DETECTED — SLOW DOWN'
                              : 'ROAD SURFACE CLEAR — NO HAZARD DETECTED')}
                        </h3>

                        <p className="text-xs sm:text-sm mt-1 leading-relaxed opacity-90">
                          {detectionResult.hazard_advisory?.alert_message}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 shrink-0 self-start md:self-auto">
                      <div className="bg-white/90 border border-slate-300 rounded-lg p-3 text-center min-w-[110px] shadow-sm">
                        <div className="text-[10px] font-bold text-slate-500 uppercase">
                          Posted Legal Limit
                        </div>
                        <div className="text-xl font-black text-slate-800 font-mono mt-0.5">
                          {detectionResult.hazard_advisory?.legal_speed_limit_kmh || legalSpeedLimit}{' '}
                          <span className="text-xs font-normal text-slate-500">km/h</span>
                        </div>
                      </div>

                      <div
                        className={`border rounded-lg p-3 text-center min-w-[120px] shadow-sm ${
                          detectionResult.potholes_detected > 0
                            ? 'bg-red-600 text-white border-red-700'
                            : 'bg-emerald-600 text-white border-emerald-700'
                        }`}
                      >
                        <div className="text-[10px] font-bold uppercase opacity-90">
                          Advisory Status
                        </div>
                        <div className="text-base font-black tracking-tight mt-0.5">
                          {detectionResult.potholes_detected > 0 ? 'SLOW DOWN' : 'MAINTAIN'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-slate-300/60 flex items-start gap-2 text-[11px] opacity-80">
                    <Info className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>
                      <strong className="font-bold">Preliminary Hazard Advisory Note:</strong> Single-image
                      inference triggers visual warning alerts. Validated scientific safe-speed computation
                      requires continuous optical distance measurement and vehicle dynamics (evaluated in video tracking).
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                  <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
                      <div className="flex items-center gap-2">
                        <Eye className="w-5 h-5 text-blue-600" />
                        <h3 className="text-base font-bold text-slate-900">
                          Detection Visualizer
                        </h3>
                      </div>

                      <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
                        <button
                          onClick={() => setViewMode('sideBySide')}
                          className={`px-2.5 py-1 rounded font-semibold transition ${
                            viewMode === 'sideBySide'
                              ? 'bg-white text-blue-700 shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Side-by-Side
                        </button>
                        <button
                          onClick={() => setViewMode('annotated')}
                          className={`px-2.5 py-1 rounded font-semibold transition ${
                            viewMode === 'annotated'
                              ? 'bg-white text-blue-700 shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Annotated
                        </button>
                        <button
                          onClick={() => setViewMode('original')}
                          className={`px-2.5 py-1 rounded font-semibold transition ${
                            viewMode === 'original'
                              ? 'bg-white text-blue-700 shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Original
                        </button>
                      </div>
                    </div>

                    {viewMode === 'sideBySide' ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="text-[11px] font-bold text-slate-500 uppercase">Original Input</div>
                          <div className="rounded-lg overflow-hidden border border-slate-300 bg-slate-900">
                            <img
                              src={imagePreview}
                              alt="Original input"
                              className="w-full h-[260px] object-contain mx-auto"
                            />
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="text-[11px] font-bold text-blue-700 uppercase">
                            Annotated YOLOv8n Output
                          </div>
                          <div className="rounded-lg overflow-hidden border border-slate-300 bg-slate-900 relative">
                            <img
                              src={detectionResult.annotated_image}
                              alt="Annotated detection"
                              className="w-full h-[260px] object-contain mx-auto"
                            />
                            <div className="absolute top-2 left-2 bg-slate-900/80 text-emerald-400 font-mono text-[10px] px-2 py-0.5 rounded border border-slate-700">
                              {detectionResult.potholes_detected} DETECTED
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="relative rounded-xl overflow-hidden border border-slate-300 bg-slate-900 shadow-inner">
                        <img
                          src={viewMode === 'annotated' ? detectionResult.annotated_image : imagePreview}
                          alt="Detection view"
                          className="w-full max-h-[440px] object-contain mx-auto"
                        />
                        <div className="absolute top-2 left-2 bg-slate-900/80 text-emerald-400 font-mono text-[11px] px-2.5 py-1 rounded border border-slate-700 flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                          <span>
                            {viewMode === 'annotated'
                              ? `YOLOv8n OUTPUT • ${detectionResult.potholes_detected} POTHOLE(S)`
                              : 'ORIGINAL INPUT IMAGE'}
                          </span>
                        </div>
                      </div>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-2">
                      <button
                        onClick={handleDownloadAnnotatedImage}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-emerald-600 text-white text-xs font-bold hover:bg-emerald-700 transition shadow-sm"
                      >
                        <Download className="w-4 h-4" />
                        Download Annotated Image
                      </button>

                      <button
                        onClick={handleReset}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-100 text-slate-700 text-xs font-bold hover:bg-slate-200 border border-slate-200 transition"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Test Another Image
                      </button>
                    </div>
                  </div>

                  <div className="lg:col-span-5 space-y-4">

                    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
                      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                        <span className="text-xs font-black text-slate-900 uppercase tracking-wide">
                          Quantitative Inference Metrics
                        </span>
                        <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                          {detectionResult.image_width} × {detectionResult.image_height} px
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 font-mono">
                        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                          <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">
                            Potholes Detected
                          </div>
                          <div
                            className={`text-2xl font-black mt-0.5 ${
                              detectionResult.potholes_detected > 0 ? 'text-red-600' : 'text-emerald-600'
                            }`}
                          >
                            {detectionResult.potholes_detected}
                          </div>
                        </div>

                        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                          <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">
                            Inference Time
                          </div>
                          <div className="text-2xl font-black text-blue-700 mt-0.5">
                            {detectionResult.inference_time_ms}{' '}
                            <span className="text-xs font-normal text-slate-500">ms</span>
                          </div>
                        </div>

                        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                          <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">
                            Average Confidence
                          </div>
                          <div className="text-lg font-bold text-slate-800 mt-0.5">
                            {detectionResult.average_confidence}
                          </div>
                        </div>

                        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                          <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">
                            Highest Confidence
                          </div>
                          <div className="text-lg font-bold text-slate-800 mt-0.5">
                            {detectionResult.highest_confidence}
                          </div>
                        </div>
                      </div>

                      <div className="text-[11px] text-slate-500 font-mono bg-slate-50 p-2.5 rounded border border-slate-200 flex justify-between">
                        <span>Threshold: {detectionResult.confidence_threshold.toFixed(2)}</span>
                        <span>Condition: {detectionResult.condition || condition}</span>
                      </div>
                    </div>

                    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                        <h4 className="text-xs font-bold text-slate-800 uppercase">
                          Detected Pothole Instances
                        </h4>
                        <span className="text-xs font-semibold text-slate-500 font-mono">
                          Count: {detectionResult.detections?.length || 0}
                        </span>
                      </div>

                      {detectionResult.detections && detectionResult.detections.length > 0 ? (
                        <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
                          {detectionResult.detections.map((det) => (
                            <div
                              key={det.id}
                              className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex flex-col gap-1.5"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <span className="w-5 h-5 rounded-full bg-red-600 text-white font-bold text-[10px] flex items-center justify-center">
                                    {det.id}
                                  </span>
                                  <span className="text-xs font-bold text-slate-800">
                                    Hazard #{det.id} (Pothole)
                                  </span>
                                </div>
                                <span className="text-xs font-mono font-black text-blue-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                                  {det.confidence}
                                </span>
                              </div>

                              <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono pt-1 border-t border-slate-200/60">
                                <span>
                                  BBox: [{det.bbox.join(', ')}]
                                </span>
                                <span>Area: {det.relative_area_pct}%</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-500">
                          <CheckCircle2 className="w-7 h-7 text-emerald-500 mx-auto mb-1.5" />
                          <div className="font-bold text-slate-700 text-sm">No Potholes Detected</div>
                          <div className="text-[11px] mt-0.5 text-slate-500">
                            Road surface appears clear of hazards at confidence threshold{' '}
                            {detectionResult.confidence_threshold.toFixed(2)}.
                          </div>
                        </div>
                      )}
                    </div>

                  </div>
                </div>

              </div>
            )}

          </div>
        )}

        {activeTab === 'review' && (
          <div className="space-y-8">
            <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-blue-600" />
                  <h2 className="text-base font-bold text-slate-800">
                    Research Methodology Flow Architecture
                  </h2>
                </div>
                <span className="text-xs font-medium text-slate-500">End-to-End Pipeline</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
                {METHODOLOGY_STEPS.map((step, idx) => (
                  <div key={step.id} className="relative flex flex-col">
                    <div className="bg-slate-50 hover:bg-blue-50/60 transition-colors border border-slate-200 hover:border-blue-300 rounded-lg p-3 text-center flex-1 flex flex-col justify-between">
                      <div>
                        <span className="inline-block w-5 h-5 rounded-full bg-blue-600 text-white text-[11px] font-bold leading-5 mb-1.5 shadow-sm">
                          {step.id}
                        </span>
                        <div className="text-[11px] font-bold text-slate-800 leading-tight uppercase">
                          {step.title}
                        </div>
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1 font-mono">
                        {step.desc}
                      </div>
                    </div>

                    {idx < METHODOLOGY_STEPS.length - 1 && (
                      <div className="hidden lg:flex absolute -right-2 top-1/2 -translate-y-1/2 z-10 text-slate-400 font-bold text-xs pointer-events-none">
                        →
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

              <div className="lg:col-span-7 space-y-6">

                <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-2 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <Eye className="w-5 h-5 text-blue-600" />
                      <h2 className="text-base font-bold text-slate-800">
                        1. Road Image Analysis (Benchmark Sample)
                      </h2>
                    </div>

                    <div className="flex items-center gap-2">
                      <input
                        type="file"
                        ref={reviewFileRef}
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) setUploadedSampleImage(URL.createObjectURL(f));
                        }}
                        accept="image/*"
                        className="hidden"
                      />
                      <button
                        onClick={() => reviewFileRef.current?.click()}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm transition"
                      >
                        <UploadCloud className="w-3.5 h-3.5" />
                        Upload Road Image
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 mb-3">
                    <span className="text-xs font-semibold text-slate-500 mr-1">Predefined Samples:</span>
                    {SAMPLES.map((sample, idx) => (
                      <button
                        key={sample.id}
                        onClick={() => {
                          setSelectedSampleIndex(idx);
                          setUploadedSampleImage(null);
                        }}
                        className={`px-3 py-1 rounded-md text-xs font-semibold transition border ${
                          !uploadedSampleImage && selectedSampleIndex === idx
                            ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                            : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                        }`}
                      >
                        {sample.shortName}
                      </button>
                    ))}
                    {uploadedSampleImage && (
                      <button
                        onClick={() => setUploadedSampleImage(null)}
                        className="px-2.5 py-1 rounded-md text-xs font-semibold bg-red-50 text-red-700 border border-red-200 hover:bg-red-100"
                      >
                        Clear Upload
                      </button>
                    )}
                  </div>

                  <RoadCanvas sample={currentSample} uploadedImage={uploadedSampleImage} />

                  <div className="mt-3 flex items-center justify-between text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-slate-700">Active Scene:</span>
                      <span>{uploadedSampleImage ? 'Custom Road Image' : currentSample.title}</span>
                    </div>
                    <div className="font-mono text-slate-600">
                      Status: <span className="text-red-600 font-bold">POTHOLE DETECTED</span> ({currentSample.confidence})
                    </div>
                  </div>
                </section>

                <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <Route className="w-5 h-5 text-emerald-600" />
                      <h2 className="text-base font-bold text-slate-800">
                        3. Safe Path Identification
                      </h2>
                    </div>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Visible-frame safe-region recommendation
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 mb-4">
                    {['left', 'center', 'right'].map((laneKey) => {
                      const reg = currentSample.regions[laneKey];
                      return (
                        <div
                          key={laneKey}
                          className={`rounded-lg p-3.5 border-2 text-center transition ${
                            reg.color === 'RED'
                              ? 'bg-red-50/80 border-red-500 text-red-900'
                              : reg.recommended
                              ? 'bg-emerald-50 border-emerald-500 text-emerald-900 ring-2 ring-emerald-400'
                              : 'bg-emerald-50/40 border-emerald-300 text-emerald-800'
                          }`}
                        >
                          <div className="text-xs font-bold font-mono tracking-wider uppercase">{laneKey} LANE</div>
                          <div className="my-1.5 flex items-center justify-center gap-1.5 font-extrabold text-sm">
                            {reg.color === 'RED' ? (
                              <>
                                <XCircle className="w-4 h-4 text-red-600" />
                                <span className="text-red-700">RED (RISK)</span>
                              </>
                            ) : (
                              <>
                                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                                <span className="text-emerald-700">GREEN</span>
                              </>
                            )}
                          </div>
                          <div className="text-[11px] font-medium opacity-80">{reg.label}</div>
                          {reg.recommended && (
                            <span className="mt-2 inline-block px-2 py-0.5 text-[10px] font-bold bg-emerald-600 text-white rounded">
                              ★ RECOMMENDED
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span className="text-xs font-bold text-emerald-900 uppercase tracking-wide">
                          SAFE PATH IDENTIFIED
                        </span>
                      </div>
                      <div className="text-sm font-extrabold text-emerald-800 mt-0.5">
                        Recommended Region: <span className="underline decoration-emerald-500">{currentSample.recommendedRegion}</span>
                      </div>
                      <p className="text-xs text-emerald-700 mt-1 font-medium">
                        Reason: {currentSample.reason}
                      </p>
                    </div>
                  </div>
                </section>
              </div>

              <div className="lg:col-span-5 space-y-6">

                <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-blue-600" />
                      <h2 className="text-base font-bold text-slate-800">
                        2. Pothole Analysis
                      </h2>
                    </div>
                    <span className="text-xs font-bold px-2.5 py-0.5 rounded-full border bg-red-100 text-red-800 border-red-300">
                      {currentSample.severity} SEVERITY
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                      <div className="text-[11px] font-semibold text-slate-500 uppercase">Detection Confidence</div>
                      <div className="text-xl font-extrabold text-blue-700 mt-0.5 font-mono">{currentSample.confidence}</div>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                      <div className="text-[11px] font-semibold text-slate-500 uppercase">Risk Score</div>
                      <div className="text-xl font-extrabold text-slate-900 mt-0.5 font-mono">{currentSample.riskScore} / 100</div>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                      <div className="text-[11px] font-semibold text-slate-500 uppercase">Estimated Length</div>
                      <div className="text-lg font-bold text-slate-800 font-mono mt-0.5">{currentSample.length}</div>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                      <div className="text-[11px] font-semibold text-slate-500 uppercase">Estimated Width</div>
                      <div className="text-lg font-bold text-slate-800 font-mono mt-0.5">{currentSample.width}</div>
                    </div>
                    <div className="col-span-2 bg-blue-50/50 rounded-lg p-3 border border-blue-200/80 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-bold text-blue-900 uppercase">Estimated Depth</div>
                        <div className="text-xl font-extrabold text-blue-800 font-mono mt-0.5">{currentSample.depth}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-amber-50/80 border border-amber-200 rounded-lg p-3 text-xs text-amber-900 flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      <span className="font-bold">Research Note:</span> “Prototype estimation. Real-world depth measurement requires calibrated camera, stereo/depth sensing, or equivalent validation.”
                    </p>
                  </div>
                </section>

                <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <Database className="w-5 h-5 text-blue-600" />
                      <h2 className="text-base font-bold text-slate-800">5. Dataset Overview</h2>
                    </div>
                    <span className="text-[11px] font-medium text-slate-500">Clean Staged Dataset</span>
                  </div>

                  <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs font-semibold text-slate-500 uppercase">Dataset Total</div>
                        <div className="text-2xl font-extrabold text-slate-900 font-mono">
                          735{' '}
                          <span className="text-xs font-normal text-slate-500">Images (1,739 Potholes)</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="px-2 py-0.5 text-[11px] font-bold rounded bg-blue-100 text-blue-800 border border-blue-200">
                          pothole
                        </span>
                        <span className="px-2 py-0.5 text-[11px] font-bold rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                          normal_road
                        </span>
                      </div>
                    </div>

                    <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden flex shadow-inner">
                      <div style={{ width: `70%` }} className="bg-blue-600 h-full" title="Train: 514"></div>
                      <div style={{ width: `20%` }} className="bg-amber-500 h-full" title="Validation: 147"></div>
                      <div style={{ width: `10%` }} className="bg-emerald-500 h-full" title="Test: 74"></div>
                    </div>
                    <div className="flex justify-between text-[11px] font-mono text-slate-500">
                      <span>Train: 70% (514)</span>
                      <span>Val: 20% (147)</span>
                      <span>Test: 10% (74)</span>
                    </div>
                  </div>
                </section>

              </div>
            </div>

            <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-5 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <h2 className="text-base font-bold text-slate-800">
                    4. Research Progress Roadmap
                  </h2>
                </div>
                <span className="text-xs font-extrabold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                  Phase 1 & 2 Completed • Segment 3 & 4 Active
                </span>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-6 space-y-4">
                  <div className="space-y-4 relative before:absolute before:left-3.5 before:top-3 before:bottom-3 before:w-0.5 before:bg-slate-200">
                    {TIMELINE_DATA.map((t) => (
                      <div key={t.stage} className="relative pl-8">
                        <div className={`absolute left-2 top-1 w-3.5 h-3.5 rounded-full border-2 border-white shadow-sm ${t.dotColor}`}></div>
                        <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-200">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-extrabold text-slate-900">{t.stage}</span>
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${t.badgeColor}`}>{t.status}</span>
                          </div>
                          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-2 gap-y-1 text-xs text-slate-700">
                            {t.items.map((item, i) => (
                              <li key={i} className="flex items-center gap-1.5">
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                                <span>{item}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="lg:col-span-6 bg-slate-50 rounded-lg p-4 border border-slate-200 space-y-3">
                  {RESEARCH_PROGRESS_ITEMS.map((item) => (
                    <div key={item.name} className="space-y-1">
                      <div className="flex justify-between text-xs font-medium text-slate-700">
                        <span className="font-semibold text-slate-800">{item.name}</span>
                        <span className="font-mono font-bold text-slate-600">{item.progress}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-2 rounded-full ${
                            item.progress === 100
                              ? 'bg-emerald-500'
                              : item.progress >= 70
                              ? 'bg-blue-600'
                              : 'bg-amber-500'
                          }`}
                          style={{ width: `${item.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </div>
        )}

      </main>

      <footer className="bg-white border-t border-slate-200 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <div>
            <span className="font-semibold text-slate-700">Project:</span> AI-Based Road Hazard Early Detection & Adaptive Safe-Speed Warning System
          </div>
          <div>
            <span className="font-semibold text-slate-700">Status:</span> Segment 4 Live Camera & Audio Warning Active
          </div>
        </div>
      </footer>
    </div>
  );
}
