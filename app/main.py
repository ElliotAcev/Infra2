from ml import train_model, predict
from db import get_db
import pandas as pd
import sys
import subprocess

def web():
    print("Starting web server...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/web.py"])
    print("Web server started.")
    
if __name__ == "__main__":
    web()
    # Initialize the database