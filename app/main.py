from TVerificacion import check_torch
import pandas as pd
import sys
import subprocess

def web():
    print("Starting web server...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/web.py"])
    print("Web server started.")
    
if __name__ == "__main__":
    print("Starting application...")
    print("Checking PyTorch environment...")
    check_torch()
    web()
    # Check PyTorch environment
    # Initialize the database