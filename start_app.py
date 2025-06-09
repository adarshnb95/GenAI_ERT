# File: start_app.py

import subprocess
import sys
import os
import signal
import time

if __name__ == "__main__":
    # 1) Optionally, ensure your env vars are loaded here or via your shell
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    # os.environ["NEWSAPI_KEY"]  = "..."

    # 2) Start Uvicorn (FastAPI) subprocess
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
        "--reload-dir", "api",
        "--reload-dir", "summarization",
    ]
    uvicorn_proc = subprocess.Popen(uvicorn_cmd)

    streamlit_cmd = [
        "streamlit", "run",
        "dashboard/dashboard_app.py",
        "--server.port", "8501"
    ]
    streamlit_proc = subprocess.Popen(streamlit_cmd)

    print(f"✅ Launched Uvicorn (pid={uvicorn_proc.pid}) and Streamlit (pid={streamlit_proc.pid})")
    print("   • Backend: http://127.0.0.1:8000/docs")
    print("   • Frontend: http://localhost:8501")

    try:
        # Poll both procs; exit if either dies or on Ctrl+C
        while True:
            if uvicorn_proc.poll() is not None:
                print("⚠️  Uvicorn has exited.")
                break
            if streamlit_proc.poll() is not None:
                print("⚠️  Streamlit has exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested. Terminating services...")
    finally:
        for p in (uvicorn_proc, streamlit_proc):
            try:
                p.terminate()
            except Exception:
                pass
