# File: start_app.py

import subprocess
import sys
import os
import signal
import time
import socket

UVICORN_HOST = "127.0.0.1"
UVICORN_PORT = 8000

def wait_for_backend(host: str, port: int, timeout: int = 30):
    """Wait until something is listening on (host, port), or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def main():
    # 1) Launch Uvicorn
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", UVICORN_HOST,
        "--port", str(UVICORN_PORT),
        "--reload",
        "--reload-dir", "api",
        "--reload-dir", "summarization",
    ]
    uvicorn_proc = subprocess.Popen(uvicorn_cmd)
    print(f"🚀 Starting FastAPI (pid={uvicorn_proc.pid})…")

    # 2) Wait for FastAPI to be ready
    print(f"🔎 Waiting up to 30s for backend at {UVICORN_HOST}:{UVICORN_PORT}…")
    if not wait_for_backend(UVICORN_HOST, UVICORN_PORT, timeout=60):
        print("❌ Backend did not appear in time; exiting.")
        uvicorn_proc.terminate()
        sys.exit(1)
    print("✅ Backend is up!")

    # 3) Launch Streamlit
    streamlit_cmd = [
        "streamlit", "run",
        "dashboard/dashboard_app.py",
        "--server.port", "8501"
    ]
    streamlit_proc = subprocess.Popen(streamlit_cmd)
    print(f"🚀 Starting Streamlit (pid={streamlit_proc.pid})…")
    print("   • Backend: http://127.0.0.1:8000/docs")
    print("   • Frontend: http://localhost:8501")

    # 4) Monitor both processes
    try:
        while True:
            if uvicorn_proc.poll() is not None:
                print("⚠️  FastAPI exited; shutting down Streamlit.")
                break
            if streamlit_proc.poll() is not None:
                print("⚠️  Streamlit exited; shutting down FastAPI.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested; terminating both services…")
    finally:
        for p in (uvicorn_proc, streamlit_proc):
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
