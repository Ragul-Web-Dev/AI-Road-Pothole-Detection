import React, { useState, useEffect, useRef, useCallback } from 'react';
import { audioAlert } from '../utils/audioAlert';
import {
  AlertTriangle,
  Camera,
  CameraOff,
  CheckCircle2,
  Circle,
  Eye,
  Gauge,
  Info,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Square,
  Volume2,
  VolumeX,
  Zap,
  Film,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

export default function LiveCameraPanel({ backendOnline, legalSpeedLimit, setLegalSpeedLimit, onRetryBackend }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [streamIntervalMs, setStreamIntervalMs] = useState(333);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [cooldownSeconds, setCooldownSeconds] = useState(2.5);
  const [cameraError, setCameraError] = useState(null);
  const [simulatedFeed, setSimulatedFeed] = useState(false);

  const [activeDetections, setActiveDetections] = useState([]);
  const [potholesCount, setPotholesCount] = useState(0);
  const [hazardAlert, setHazardAlert] = useState(null);
  const [inferenceLatencyMs, setInferenceLatencyMs] = useState(0);
  const [liveFps, setLiveFps] = useState(0);
  const [lastAudioTriggerTime, setLastAudioTriggerTime] = useState(0);
  const [frameCount, setFrameCount] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const loopTimerRef = useRef(null);
  const isProcessingRef = useRef(false);
  const isStreamingRef = useRef(false);
  const fpsTrackerRef = useRef({ lastTime: Date.now(), frames: 0 });
  const simulatedIndexRef = useRef(0);

  useEffect(() => {
    audioAlert.setMuted(isMuted);
  }, [isMuted]);

  useEffect(() => {
    audioAlert.setCooldown(cooldownSeconds * 1000);
  }, [cooldownSeconds]);

  useEffect(() => {
    const getDevices = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoInputs = devices.filter((d) => d.kind === 'videoinput');
          setCameraDevices(videoInputs);
          if (videoInputs.length > 0 && !selectedDeviceId) {
            setSelectedDeviceId(videoInputs[0].deviceId);
          }
        }
      } catch (err) {
        console.warn('Unable to enumerate camera devices:', err);
      }
    };
    getDevices();
  }, []);

  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, []);

  const startCameraStream = async () => {
    setCameraError(null);
    audioAlert.init();

    if (simulatedFeed) {
      isStreamingRef.current = true;
      setIsStreaming(true);
      startInferenceLoop();
      return;
    }

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access API is not supported in this browser environment.');
      }

      const constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      isStreamingRef.current = true;
      setIsStreaming(true);
      startInferenceLoop();
    } catch (err) {
      console.error('Camera startup error:', err);
      let errorMsg = 'Failed to access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMsg = 'Camera permission was denied. Please allow camera access in your browser settings, or use the Simulated Stream mode.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMsg = 'No camera device detected on this system. You can switch to the Simulated Stream mode below.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMsg = 'Camera is already in use by another application or tab.';
      } else {
        errorMsg = `Camera Error: ${err.message || err.name}`;
      }
      setCameraError(errorMsg);
      stopCameraStream();
    }
  };

  const stopCameraStream = useCallback(() => {
    isStreamingRef.current = false;
    setIsStreaming(false);

    if (loopTimerRef.current) {
      clearTimeout(loopTimerRef.current);
      loopTimerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setActiveDetections([]);
    setPotholesCount(0);
    setHazardAlert(null);
    clearOverlayCanvas();
  }, []);

  const clearOverlayCanvas = () => {
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
  };

  const drawDetections = (detections, imgW, imgH) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections || detections.length === 0) return;

    const scaleX = canvas.width / (imgW || 640);
    const scaleY = canvas.height / (imgH || 480);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const dx1 = x1 * scaleX;
      const dy1 = y1 * scaleY;
      const dx2 = x2 * scaleX;
      const dy2 = y2 * scaleY;
      const dw = dx2 - dx1;
      const dh = dy2 - dy1;

      const confVal = det.confidence_val || 0.5;
      const strokeColor = confVal >= 0.7 ? '#ef4444' : '#f97316';
      const fillColor = confVal >= 0.7 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(249, 115, 22, 0.15)';

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(dx1, dy1, dw, dh);

      ctx.fillStyle = fillColor;
      ctx.fillRect(dx1, dy1, dw, dh);

      const labelText = `POTHOLE ${det.confidence}`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const textHeight = 14;

      const pillY = Math.max(0, dy1 - textHeight - 6);
      ctx.fillStyle = strokeColor;
      ctx.beginPath();
      ctx.roundRect(dx1, pillY, textWidth + 12, textHeight + 6, 4);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, dx1 + 6, pillY + textHeight);
    });
  };

  const runSingleInferenceFrame = async () => {
    if (!isStreamingRef.current || isProcessingRef.current) {
      return;
    }

    isProcessingRef.current = true;
    const captureCanvas = canvasRef.current;
    const video = videoRef.current;

    let base64Image = '';

    if (!simulatedFeed && video && video.readyState >= 2 && captureCanvas) {
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      captureCanvas.width = width;
      captureCanvas.height = height;
      const ctx = captureCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);
      base64Image = captureCanvas.toDataURL('image/jpeg', 0.82);
    } else if (simulatedFeed) {
      const simulatedImages = [
        'dataset/images/test/img-104.jpg',
        'dataset/images/test/img-121.jpg',
        'dataset/images/test/img-126.jpg',
        'dataset/images/test/normal_road_001.jpg',
        'dataset/images/test/img-128.jpg',
        'dataset/images/test/normal_road_028.jpg',
      ];
      const curFilename = simulatedImages[simulatedIndexRef.current % simulatedImages.length];
      simulatedIndexRef.current += 1;

      try {
        const sampleRes = await fetch(`${API_BASE_URL}/api/samples`);
        const sampleData = await sampleRes.json();
        const found = sampleData.samples?.find((s) => s.filename.includes('104') || s.filename.includes('121')) || sampleData.samples?.[0];
        if (found) {
          base64Image = found.data_url;
        }
      } catch {
      }
    }

    if (!base64Image) {
      isProcessingRef.current = false;
      return;
    }

    const tStart = performance.now();

    try {
      const response = await fetch(`${API_BASE_URL}/api/detect/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Image,
          conf: confidenceThreshold,
          legal_speed_limit: legalSpeedLimit,
          filename: simulatedFeed ? 'simulated_dashcam_frame.jpg' : 'live_camera_frame.jpg',
          condition: 'Live Stream',
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const totalLatency = Math.round(performance.now() - tStart);

        setActiveDetections(data.detections || []);
        setPotholesCount(data.potholes_detected || 0);
        setHazardAlert(data.hazard_advisory || null);
        setInferenceLatencyMs(data.inference_time_ms || totalLatency);
        setFrameCount((prev) => prev + 1);

        const now = Date.now();
        fpsTrackerRef.current.frames += 1;
        if (now - fpsTrackerRef.current.lastTime >= 1000) {
          setLiveFps(fpsTrackerRef.current.frames);
          fpsTrackerRef.current.frames = 0;
          fpsTrackerRef.current.lastTime = now;
        }

        if (data.potholes_detected > 0) {
          const played = audioAlert.playHazardBeep();
          if (played) {
            setLastAudioTriggerTime(Date.now());
          }
        }

        drawDetections(data.detections || [], data.image_width || 640, data.image_height || 480);
      }
    } catch (err) {
      console.warn('Live inference loop error:', err);
    } finally {
      isProcessingRef.current = false;

      if (isStreamingRef.current) {
        loopTimerRef.current = setTimeout(runSingleInferenceFrame, streamIntervalMs);
      }
    }
  };

  const startInferenceLoop = () => {
    if (loopTimerRef.current) clearTimeout(loopTimerRef.current);
    runSingleInferenceFrame();
  };

  const handleTestBeep = () => {
    audioAlert.init();
    audioAlert.playHazardBeep(true);
  };

  return (
    <div className="space-y-6">
      {!backendOnline && (
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 text-xs text-amber-900 flex flex-col sm:flex-row sm:items-start justify-between gap-3 shadow-sm">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-amber-950">FastAPI ML Backend Service Offline</div>
              <div className="mt-0.5 text-amber-800">
                Live camera inference requires the FastAPI backend to be running with the trained YOLOv8n checkpoint.
              </div>
              <div className="mt-2 text-[12px] font-mono bg-amber-100 p-2 rounded border border-amber-300 select-all">
                python pothole-research/scripts/server.py --port 5000
              </div>
            </div>
          </div>
          {onRetryBackend && (
            <button
              onClick={onRetryBackend}
              className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold transition shrink-0 inline-flex items-center gap-1.5 self-start"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry Connection
            </button>
          )}
        </div>
      )}

      {cameraError && (
        <div className="bg-red-50 border border-red-300 rounded-xl p-4 text-xs text-red-900 flex items-start justify-between gap-3 shadow-sm">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-red-950">Camera Error</div>
              <div className="mt-0.5">{cameraError}</div>
              <div className="mt-2">
                <button
                  onClick={() => {
                    setSimulatedFeed(true);
                    setCameraError(null);
                  }}
                  className="px-3 py-1 rounded bg-red-600 hover:bg-red-700 text-white font-bold transition inline-flex items-center gap-1.5"
                >
                  <Film className="w-3.5 h-3.5" />
                  Switch to Simulated Stream Mode
                </button>
              </div>
            </div>
          </div>
          <button onClick={() => setCameraError(null)} className="text-red-500 hover:text-red-700 font-bold">
            ✕
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-slate-900">
                Live Video Feed & Hazard HUD Overlay
              </h3>
            </div>

            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-bold ${
                  isStreaming
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-300 animate-pulse'
                    : 'bg-slate-100 text-slate-600 border border-slate-200'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-emerald-500' : 'bg-slate-400'}`}></span>
                {isStreaming ? (simulatedFeed ? 'SIMULATED STREAM LIVE' : 'CAMERA LIVE') : 'FEED STANDBY'}
              </span>

              {isStreaming && (
                <span className="text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded border border-blue-200">
                  {liveFps > 0 ? `${liveFps} FPS` : 'Measuring...'}
                </span>
              )}
            </div>
          </div>

          <div className="relative rounded-xl overflow-hidden border border-slate-300 bg-slate-950 aspect-[4/3] sm:aspect-video flex items-center justify-center shadow-inner group">
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-contain ${!isStreaming || simulatedFeed ? 'hidden' : 'block'}`}
            />

            {!isStreaming && (
              <div className="text-center p-6 space-y-3">
                <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-700 text-slate-400 flex items-center justify-center mx-auto">
                  <CameraOff className="w-8 h-8" />
                </div>
                <div>
                  <div className="text-sm font-bold text-white">Live Camera Standby</div>
                  <div className="text-xs text-slate-400 mt-1 max-w-sm">
                    Click <strong>Start Camera</strong> to capture live frames for real-time YOLOv8n road hazard detection.
                  </div>
                </div>
              </div>
            )}

            <canvas
              ref={overlayCanvasRef}
              width={640}
              height={480}
              className="absolute inset-0 w-full h-full pointer-events-none z-10"
            />

            <canvas ref={canvasRef} className="hidden" />

            {isStreaming && (
              <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2">
                <div className="bg-slate-950/80 backdrop-blur text-emerald-400 font-mono text-[11px] px-2.5 py-1 rounded border border-slate-700 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>YOLOv8n ACTIVE</span>
                </div>
                <div className="bg-slate-950/80 backdrop-blur text-white font-mono text-[11px] px-2 py-1 rounded border border-slate-700">
                  Latency: <span className="text-blue-400 font-bold">{inferenceLatencyMs} ms</span>
                </div>
              </div>
            )}

            {isStreaming && (
              <div className="absolute top-3 right-3 z-20">
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className={`p-2 rounded-lg backdrop-blur text-xs font-bold border transition flex items-center gap-1.5 shadow ${
                    isMuted
                      ? 'bg-red-600/90 hover:bg-red-700 text-white border-red-500'
                      : 'bg-slate-950/80 hover:bg-slate-900 text-emerald-400 border-slate-700'
                  }`}
                  title={isMuted ? 'Unmute Audio Warning Beep' : 'Mute Audio Warning Beep'}
                >
                  {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  <span>{isMuted ? 'MUTED' : 'AUDIO ON'}</span>
                </button>
              </div>
            )}

            {isStreaming && (
              <div className="absolute bottom-3 inset-x-3 z-20">
                <div
                  className={`rounded-lg p-3 backdrop-blur border transition flex items-center justify-between gap-3 shadow-lg ${
                    potholesCount > 0
                      ? 'bg-red-950/90 border-red-500 text-white animate-pulse'
                      : 'bg-slate-950/85 border-slate-700 text-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    {potholesCount > 0 ? (
                      <ShieldAlert className="w-6 h-6 text-red-500 shrink-0 animate-bounce" />
                    ) : (
                      <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0" />
                    )}
                    <div>
                      <div className="text-xs font-black tracking-tight uppercase">
                        {potholesCount > 0
                          ? 'POTHOLE DETECTED — SLOW DOWN'
                          : 'ROAD CLEAR — NO HAZARDS DETECTED'}
                      </div>
                      <div className="text-[11px] opacity-85 font-sans">
                        {potholesCount > 0
                          ? `Visual hazard identified (${potholesCount} pothole${potholesCount > 1 ? 's' : ''}). Reduce speed below ${legalSpeedLimit} km/h.`
                          : `Normal road surface clear. Legal limit: ${legalSpeedLimit} km/h.`}
                      </div>
                    </div>
                  </div>

                  <div className="text-right shrink-0 font-mono">
                    <div className="text-[10px] uppercase opacity-75">Posted Limit</div>
                    <div className="text-sm font-black text-white">{legalSpeedLimit} km/h</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <div className="flex items-center gap-2">
              {!isStreaming ? (
                <button
                  onClick={startCameraStream}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-black uppercase tracking-wider transition shadow-md shadow-blue-500/20 active:scale-[0.99]"
                >
                  <Play className="w-4 h-4 fill-white" />
                  Start Live Camera
                </button>
              ) : (
                <button
                  onClick={stopCameraStream}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-black uppercase tracking-wider transition shadow-md shadow-red-500/20 active:scale-[0.99]"
                >
                  <Square className="w-4 h-4 fill-white" />
                  Stop Camera Stream
                </button>
              )}

              <button
                onClick={handleTestBeep}
                className="inline-flex items-center gap-1.5 px-3 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold border border-slate-200 transition"
                title="Trigger a test audio alert tone"
              >
                <Volume2 className="w-4 h-4 text-blue-600" />
                Test Beep
              </button>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <label className="text-slate-500 font-semibold">Feed Source:</label>
              <button
                onClick={() => {
                  if (isStreaming) stopCameraStream();
                  setSimulatedFeed(false);
                }}
                className={`px-2.5 py-1 rounded-lg font-bold border transition ${
                  !simulatedFeed
                    ? 'bg-blue-50 text-blue-700 border-blue-300'
                    : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                }`}
              >
                Webcam
              </button>
              <button
                onClick={() => {
                  if (isStreaming) stopCameraStream();
                  setSimulatedFeed(true);
                }}
                className={`px-2.5 py-1 rounded-lg font-bold border transition ${
                  simulatedFeed
                    ? 'bg-blue-50 text-blue-700 border-blue-300'
                    : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                }`}
              >
                Simulated Loop
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <span className="text-xs font-black text-slate-900 uppercase tracking-wide flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-blue-600" />
                Live Camera Configuration
              </span>
            </div>

            {cameraDevices.length > 1 && !simulatedFeed && (
              <div>
                <label className="text-[11px] font-bold text-slate-700 uppercase mb-1 block">
                  Select Camera Device
                </label>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  disabled={isStreaming}
                  className="w-full text-xs font-mono p-2 rounded-lg border border-slate-200 bg-slate-50"
                >
                  {cameraDevices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800 uppercase flex items-center gap-1">
                  <Gauge className="w-3.5 h-3.5 text-blue-600" />
                  Posted Legal Speed
                </label>
                <span className="text-xs font-mono font-black text-blue-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                  {legalSpeedLimit} km/h
                </span>
              </div>
              <input
                type="range"
                min="20"
                max="120"
                step="5"
                value={legalSpeedLimit}
                onChange={(e) => setLegalSpeedLimit(parseInt(e.target.value, 10))}
                className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
              />
              <div className="flex justify-between gap-1">
                {[30, 40, 50, 60, 80].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setLegalSpeedLimit(s)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border transition ${
                      legalSpeedLimit === s
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800 uppercase">
                  Frame Sample Rate
                </label>
                <span className="text-xs font-mono font-bold text-slate-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                  {streamIntervalMs === 200 ? '5 FPS (200ms)' : streamIntervalMs === 333 ? '3 FPS (333ms)' : streamIntervalMs === 500 ? '2 FPS (500ms)' : '1 FPS (1000ms)'}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-1">
                {[
                  { ms: 1000, label: '1 FPS' },
                  { ms: 500, label: '2 FPS' },
                  { ms: 333, label: '3 FPS' },
                  { ms: 200, label: '5 FPS' },
                ].map((rate) => (
                  <button
                    key={rate.ms}
                    type="button"
                    onClick={() => setStreamIntervalMs(rate.ms)}
                    className={`py-1 rounded text-[11px] font-mono font-bold border transition ${
                      streamIntervalMs === rate.ms
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {rate.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-500">
                * Prevents overlapping requests by waiting for current frame completion before dispatching next.
              </p>
            </div>

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800 uppercase">
                  Confidence Threshold
                </label>
                <span className="text-xs font-mono font-black text-blue-700 bg-white px-2 py-0.5 rounded border border-slate-200">
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
            </div>

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800 uppercase">
                  Audio Warning Cooldown
                </label>
                <span className="text-xs font-mono font-bold text-slate-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                  {cooldownSeconds}s
                </span>
              </div>
              <input
                type="range"
                min="1.0"
                max="5.0"
                step="0.5"
                value={cooldownSeconds}
                onChange={(e) => setCooldownSeconds(parseFloat(e.target.value))}
                className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
              />
              <div className="text-[10px] text-slate-500">
                * Prevents continuous piercing beeps on consecutive hazard frames.
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <h4 className="text-xs font-black text-slate-900 uppercase tracking-wide">
                Live Frame Telemetry
              </h4>
              <span className="text-xs font-mono text-slate-500">
                Frames Evaluated: {frameCount}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2.5 font-mono">
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">Potholes</div>
                <div
                  className={`text-xl font-black mt-0.5 ${
                    potholesCount > 0 ? 'text-red-600' : 'text-emerald-600'
                  }`}
                >
                  {potholesCount}
                </div>
              </div>

              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-500 uppercase font-sans font-bold">Inference</div>
                <div className="text-xl font-black text-blue-700 mt-0.5">
                  {inferenceLatencyMs} <span className="text-xs font-normal text-slate-500">ms</span>
                </div>
              </div>
            </div>

            <div className="space-y-1.5 pt-1">
              <div className="text-[11px] font-bold text-slate-700 uppercase">
                Active Detections in Frame:
              </div>
              {activeDetections.length > 0 ? (
                <div className="space-y-1.5 max-h-[140px] overflow-y-auto pr-1">
                  {activeDetections.map((det) => (
                    <div
                      key={det.id}
                      className="bg-red-50/80 border border-red-200 rounded p-2 flex items-center justify-between text-xs"
                    >
                      <span className="font-bold text-red-900">Hazard #{det.id} (Pothole)</span>
                      <span className="font-mono font-bold text-red-700 bg-white px-1.5 py-0.5 rounded border border-red-200">
                        {det.confidence}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-slate-400 bg-slate-50 p-2 rounded text-center border border-slate-200 font-mono">
                  No potholes currently in view
                </div>
              )}
            </div>

            <div className="bg-blue-50/60 border border-blue-200 rounded-lg p-3 text-[11px] text-blue-900 flex items-start gap-2">
              <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
              <div className="leading-tight">
                <strong>Preliminary Advisory Note:</strong> Live warning alert triggers audio and visual advisories. Precise safe-speed calculation requires multi-frame optical distance estimation and vehicle telemetry.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
