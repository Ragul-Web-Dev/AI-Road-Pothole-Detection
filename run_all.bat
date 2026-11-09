@echo off
echo ========================================================
echo   Real-Time Road Pothole Detection & Driver Warning
echo ========================================================
echo.
echo 1. Starting FastAPI ML Backend on Port 5000...
start "Backend - FastAPI YOLOv8n (Port 5000)" cmd /k "python scripts/server.py --port 5000"

echo 2. Starting React Vite Frontend on Port 5173...
start "Frontend - React Vite (Port 5173)" cmd /k "npm run dev"

echo 3. Opening application in browser...
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo Both servers are now running!
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:5000
echo ========================================================
