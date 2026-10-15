# AI Smart Farming Assistant

**Level:** Intermediate | **Stack:** Python, Streamlit, pandas, scikit-learn

A learning-focused web app that accepts farm and soil inputs, produces an ML-based demo crop suggestion, and gives simple rule-based nutrient and irrigation awareness.

## Important limitation
`data/demo_crop_data.csv` is **synthetic demonstration data**, created to make the project runnable offline. The model score is only a score on held-out synthetic data. It is not evidence of field accuracy. Do not use this app as the sole basis for crop, fertilizer, or irrigation decisions. Validate with soil testing and local agriculture experts.

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt/PowerShell inside the `AI_Smart_Farming_Assistant` folder.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it:
   ```bash
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # If using Command Prompt:
   .venv\Scripts\activate.bat
   ```
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
7. Launch:
   ```bash
   streamlit run app.py
   ```
8. Browser opens the local app (usually `http://localhost:8501`).

## Run on Linux / WSL
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Input N, P, K, soil pH, rainfall, temperature, humidity, and soil moisture.
- Random Forest classifier trained from bundled demo data.
- Top 3 model outputs and an explicitly labelled demo score.
- Rule-based nutrient checks and irrigation awareness.
- Download farm inputs as CSV.
- Responsible-use notice.

## Project structure
```text
AI_Smart_Farming_Assistant/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── demo_crop_data.csv
└── docs/
    ├── lean_canvas.md
    ├── concept_note.md
    └── ppt_content.md
```

## How the ML part works
1. Load the CSV with seven input features and a crop label.
2. Split rows into train and test subsets.
3. Fit a scikit-learn pipeline with median imputation and Random Forest classification.
4. Calculate a held-out score on the synthetic test split.
5. Fit the model on all bundled rows for demo predictions.

## Suggested future improvements
- Replace synthetic examples with a licensed, documented, locally relevant agronomic dataset.
- Add district-specific crop calendars and crop-stage logic reviewed by agricultural experts.
- Add optional live weather API with clear source/time stamps.
- Add multilingual UI (Hindi/Rajasthani).
- Evaluate using field data and precision/recall per crop before any real-world deployment.
