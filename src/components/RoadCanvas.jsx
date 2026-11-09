import React from 'react';

export default function RoadCanvas({ sample, uploadedImage }) {
  if (uploadedImage) {
    return (
      <div className="relative w-full aspect-[16/10] bg-slate-900 rounded-lg overflow-hidden border border-slate-300 shadow-inner group">
        <img
          src={uploadedImage}
          alt="Uploaded road input"
          className="w-full h-full object-cover"
        />

        <div
          className="absolute border-2 border-red-500 bg-red-500/20 rounded shadow-lg transition-all duration-300 pointer-events-none"
          style={{
            left: sample.box.x,
            top: sample.box.y,
            width: sample.box.width,
            height: sample.box.height,
          }}
        >
          <div className="absolute -top-7 left-0 bg-red-600 text-white font-mono text-xs font-semibold px-2 py-0.5 rounded shadow flex items-center gap-1.5 whitespace-nowrap">
            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
            <span>POTHOLE</span>
            <span className="bg-red-800/80 px-1 py-0.2 text-[10px] rounded">
              {sample.confidence}
            </span>
          </div>

          <div className="absolute -bottom-6 left-0 bg-slate-900/90 text-amber-300 font-mono text-[11px] px-1.5 py-0.5 rounded border border-slate-700">
            Est: {sample.length} × {sample.width} (d: {sample.depth})
          </div>

          <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white"></div>
          <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white"></div>
          <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white"></div>
          <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white"></div>
        </div>

        <div className="absolute top-2 left-3 font-mono text-[11px] text-slate-300/80 bg-slate-900/70 px-2 py-0.5 rounded border border-slate-700/50 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>CAM_01: CUSTOM_UPLOAD (1920x1080)</span>
        </div>
      </div>
    );
  }

  const potholeCoords = {
    center: { cx: 400, cy: 300, rx: 75, ry: 40, innerRx: 60, innerRy: 30 },
    left: { cx: 240, cy: 310, rx: 60, ry: 35, innerRx: 45, innerRy: 24 },
    right: { cx: 560, cy: 320, rx: 45, ry: 25, innerRx: 32, innerRy: 16 },
  }[sample.svgType] || { cx: 400, cy: 300, rx: 75, ry: 40, innerRx: 60, innerRy: 30 };

  return (
    <div className="relative w-full aspect-[16/10] bg-slate-900 rounded-lg overflow-hidden border border-slate-300 shadow-inner group select-none">
      <svg
        viewBox="0 0 800 500"
        className="w-full h-full object-cover"
        preserveAspectRatio="none"
      >
        <defs>
          <pattern id="asphalt" width="40" height="40" patternUnits="userSpaceOnUse">
            <rect width="40" height="40" fill="#2d3748" />
            <circle cx="10" cy="10" r="1.5" fill="#1a202c" opacity="0.6" />
            <circle cx="30" cy="15" r="2" fill="#4a5568" opacity="0.5" />
            <circle cx="20" cy="30" r="1.2" fill="#1a202c" opacity="0.7" />
            <circle cx="35" cy="35" r="1.8" fill="#4a5568" opacity="0.4" />
          </pattern>

          <radialGradient id="potholeDepth" cx="45%" cy="40%" r="55%">
            <stop offset="0%" stopColor="#0f172a" />
            <stop offset="60%" stopColor="#1e293b" />
            <stop offset="85%" stopColor="#334155" />
            <stop offset="100%" stopColor="#475569" />
          </radialGradient>

          <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#94a3b8" />
            <stop offset="100%" stopColor="#cbd5e1" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width="800" height="150" fill="url(#skyGrad)" />

        <path d="M0,150 L120,135 L280,145 L450,130 L600,140 L800,135 L800,150 Z" fill="#64748b" />
        <path d="M0,150 L200,142 L500,140 L800,144 L800,150 Z" fill="#475569" />

        <polygon points="260,150 540,150 800,500 0,500" fill="url(#asphalt)" />

        <polygon points="230,150 260,150 0,500 0,500" fill="#334155" />
        <line x1="260" y1="150" x2="0" y2="500" stroke="#fbbf24" strokeWidth="3" strokeDasharray="15,10" />

        <polygon points="540,150 570,150 800,500 800,500" fill="#334155" />
        <line x1="540" y1="150" x2="800" y2="500" stroke="#ffffff" strokeWidth="4" />

        <line x1="400" y1="150" x2="400" y2="500" stroke="#f8fafc" strokeWidth="4" strokeDasharray="25,20" />

        <line x1="330" y1="150" x2="200" y2="500" stroke="#64748b" strokeWidth="1" strokeDasharray="6,6" opacity="0.5" />
        <line x1="470" y1="150" x2="600" y2="500" stroke="#64748b" strokeWidth="1" strokeDasharray="6,6" opacity="0.5" />

        <g id="pothole-object">
          <ellipse
            cx={potholeCoords.cx}
            cy={potholeCoords.cy}
            rx={potholeCoords.rx}
            ry={potholeCoords.ry}
            fill="#1e293b"
            stroke="#0f172a"
            strokeWidth="3"
          />

          <ellipse
            cx={potholeCoords.cx - 2}
            cy={potholeCoords.cy + 3}
            rx={potholeCoords.innerRx}
            ry={potholeCoords.innerRy}
            fill="url(#potholeDepth)"
          />

          <path
            d={`M${potholeCoords.cx - potholeCoords.innerRx + 8},${potholeCoords.cy} 
                Q${potholeCoords.cx},${potholeCoords.cy + potholeCoords.innerRy - 4} 
                ${potholeCoords.cx + potholeCoords.innerRx - 10},${potholeCoords.cy + 2}
                Q${potholeCoords.cx + 5},${potholeCoords.cy + potholeCoords.innerRy + 6}
                ${potholeCoords.cx - potholeCoords.innerRx + 8},${potholeCoords.cy}`}
            fill="#050811"
            opacity="0.9"
          />

          <path
            d={`M${potholeCoords.cx - potholeCoords.rx},${potholeCoords.cy} 
                l-18,-8 l-10,4 M${potholeCoords.cx + potholeCoords.rx},${potholeCoords.cy - 5} 
                l15,-5 l12,9 M${potholeCoords.cx + 10},${potholeCoords.cy + potholeCoords.ry} 
                l8,16 l-6,10`}
            stroke="#0f172a"
            strokeWidth="1.8"
            fill="none"
            opacity="0.75"
          />
        </g>
      </svg>

      <div
        className="absolute border-2 border-red-500 bg-red-500/20 rounded shadow-lg transition-all duration-300 pointer-events-none"
        style={{
          left: sample.box.x,
          top: sample.box.y,
          width: sample.box.width,
          height: sample.box.height,
        }}
      >
        <div className="absolute -top-7 left-0 bg-red-600 text-white font-mono text-xs font-semibold px-2 py-0.5 rounded shadow flex items-center gap-1.5 whitespace-nowrap">
          <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
          <span>{sample.box.label}</span>
          <span className="bg-red-800/80 px-1 py-0.2 text-[10px] rounded">
            {sample.confidence}
          </span>
        </div>

        <div className="absolute -bottom-6 left-0 bg-slate-900/90 text-amber-300 font-mono text-[11px] px-1.5 py-0.5 rounded border border-slate-700">
          Est: {sample.length} × {sample.width} (d: {sample.depth})
        </div>

        <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white"></div>
        <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white"></div>
        <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white"></div>
        <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white"></div>
      </div>

      <div className="absolute top-2 left-3 font-mono text-[11px] text-slate-300/80 bg-slate-900/70 px-2.5 py-1 rounded border border-slate-700/50 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span>RGB CAMERA FEED: {sample.id.toUpperCase()}</span>
      </div>

      <div className="absolute top-2 right-3 font-mono text-[10px] text-slate-300/70 bg-slate-900/70 px-2 py-0.5 rounded border border-slate-700/50">
        1280 × 720 px | 30 FPS
      </div>
    </div>
  );
}
