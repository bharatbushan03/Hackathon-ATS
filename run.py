"""Start API (uvicorn) and UI (streamlit) together:  python run.py"""
import subprocess
import sys

procs = [
    subprocess.Popen([sys.executable, "-m", "uvicorn", "hackeval.api:app", "--host", "127.0.0.1", "--port", "8000"]),
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]),
]
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    pass
finally:
    for p in procs:
        p.terminate()
