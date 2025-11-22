"""
Quick launcher for the Streamlit web interface
"""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    
    print("=" * 60)
    print("Starting Hybrid CDS Web Interface...")
    print("=" * 60)
    print("\nThe web interface will open in your browser.")
    print("If it doesn't open automatically, go to:")
    print("  http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")

if __name__ == "__main__":
    main()

