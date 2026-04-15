# Bank Term Deposit Subscription Predictor

This project predicts whether a bank customer is likely to subscribe to a term deposit using the UCI Bank Marketing dataset.

## Files
- `train.py` trains the models and saves the artifacts.
- `app.py` runs the Streamlit app.
- `artifacts/` contains the trained model, metrics, charts, and summaries.

## Setup
```bash
python -m venv .venv
```

### Activate on Windows
```bash
.venv\Scripts\activate
```

### Activate on macOS/Linux
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python train.py Or run the train.ipynb notebook
streamlit run app.py
```

## Notes
The feature `duration` is excluded because it is only known after the call ends and would make the system unrealistic for real deployment.
