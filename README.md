# NeuralHydrology Model Evaluation and Input Data Validation

## Installation & Setup
1. **Clone the repo**  
 ```bash
 git clone https://github.com/fai05/model-eval-and-data-validator.git
 
 cd model-eval-and-data-validator 
```

2. **(Optional) Create & activate a virtual environment**
 ```bash
  python -m venv venv
  ```
  *(Windows)*
  ```bash
  venv\Scripts\activate
  ```
  *(macOS/Linux)*
  ```bash
  source venv/bin/activate
```
3. Install dependencies
```bash
  pip install -r requirements.txt
```
4. Running the App
```bash
  streamlit run discharge_tool.py
```
---
## Features

### 1. Discharge Evaluation Tool
- **Upload** predicted & observed discharge (CSV or Pickle)
- **Automatic** frequency detection (hourly/daily) & resampling
- **Performance metrics**: MAE, RMSE, R², NSE, KGE, PBIAS, Peak‑Timing
- **Interactive plots**:  
  • 1:1 Scatter  
  • Hydrograph  
  • Residuals  
  • Histogram  
  • Flow‑Duration Curve  
  • Seasonal Error  

### 2. Input Data Validator
- **Upload** NetCDF hydrological inputs
- **Date‑range** filtering
- **Flags** for missing values, flatlines, spikes, out‑of‑range & long constant periods
- **Quality scores** (with/without spikes)
- **Visualizations**: monthly bar charts, heatmaps, time‑series with issue markers
- **Download** flagged records as CSV

---

## Requirements

- Python 3.8+
- See `requirements.txt`

---
