# SpikeSense - Real-time Churn Spike Detector 🎯

This project detects churn spikes using a machine learning model and visualizes anomalies via a web interface built with Streamlit.

## How to Run

1. Clone the repo or unzip the folder.
2. Make sure Python is installed.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run streamlit_churn_spike_detector.py
```

5. Upload a CSV file or use the default `sample_churn_data.csv`

## Expected Format

CSV must include a binary column `is_churn` with 0 (not churned) and 1 (churned).