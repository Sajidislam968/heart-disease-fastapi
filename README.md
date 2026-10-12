# Heart Disease Prediction API (FastAPI + Docker + Render)

A small end-to-end machine-learning deployment assignment. It trains a Logistic Regression classifier on the Heart Disease dataset, saves the fitted scikit-learn pipeline with `joblib`, serves predictions through FastAPI, and packages the app with Docker / Docker Compose.

> **Educational use only.** The output is a prediction of the dataset label, not a medical diagnosis or a substitute for professional care.

## Features

- `POST /predict`: validates the 13 dataset input features and returns a boolean `heart_disease` result plus the predicted class and probability.
- `GET /health`: reports whether the trained model is available.
- `GET /info`: model details, dataset information, target labels, and feature names.
- FastAPI Swagger UI at `/docs`.
- `Dockerfile` and `docker-compose.yml`.
- Deterministic 80/20 train/test split, training metrics saved to `model/metadata.json`.
- `render.yaml` Blueprint starter configuration.

## Repository structure

```text
heart-disease-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   └── .gitkeep
├── model/
│   └── .gitkeep
├── scripts/
│   └── train_model.py
├── tests/
│   ├── test_api.py
│   └── test_schemas.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
└── README.md
```

The trained model (`model/heart_model.joblib`) and model metrics (`model/metadata.json`) are generated when you run the training script. The Docker build runs this script automatically, so the final image contains the model. They are not checked into this starter repository because the current build environment could not download the source CSV; the script downloads it in an environment with internet access.

## Dataset

- Assignment dataset: [Kaggle — Heart Disease Dataset by johnsmith88](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Public CSV mirror used by the training script when downloading automatically: [heart.csv](https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv)

The expected target column is `target`, where `0` means absence and `1` means presence in this dataset copy. Input feature order is `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

## Run locally without Docker

Python 3.11 is recommended.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/train_model.py --download-if-missing
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).

If you prefer downloading the Kaggle CSV manually, save it as `data/heart.csv` and run:

```bash
python scripts/train_model.py
```

If downloading a public mirror fails because the URL changes or the network is restricted, manual download from Kaggle is the fallback. Do not place Kaggle credentials in source code.

## Run with Docker Compose

Make sure Docker Engine and the Docker Compose plugin are installed and running.

```bash
docker compose build
docker compose up
```

The image build installs dependencies, downloads the dataset if `data/heart.csv` is absent, trains the classifier, and saves the joblib artifact inside the image. Visit [http://localhost:8000/docs](http://localhost:8000/docs).
<img width="1142" height="381" alt="image" src="https://github.com/user-attachments/assets/5d0e7014-0c4c-4679-8690-af986703acf0" />


Run detached and view logs:

```bash
docker compose up -d
docker compose logs -f
```

Stop the container:

```bash
docker compose down
```

## API examples

### `GET /health`

```bash
curl http://localhost:8000/health
```

Example response:

```json
{"status":"ok","model_loaded":true}
```

### `GET /info`

```bash
curl http://localhost:8000/info
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

The exact prediction depends on the model trained from the dataset. Response shape:

```json
{
  "heart_disease": false,
  "prediction": "absence",
  "probability": 0.31,
  "model": "StandardScaler + LogisticRegression",
  "disclaimer": "Educational demo only; not a medical diagnosis."
}
```

The values above illustrate the response schema only; they are not a guaranteed prediction for the example patient.

## Test

```bash
pytest -q
```

The unit tests validate request schema constraints and API response behavior with a test double. For an end-to-end prediction test using the actual trained model, train the model first and send the `/predict` example above.

## Deploy to Render (Docker)

1. Push this project to a new GitHub repository.
2. In the [Render Dashboard](https://dashboard.render.com/), choose **New → Web Service** and connect the repository.
3. Choose **Docker** as the runtime/environment. Keep the Dockerfile and `render.yaml` at repository root (build context `.` / root directory).
4. Set the health check path to `/health` if it is not picked up from the Blueprint. Render provides the service's `PORT`; this app binds to `0.0.0.0` and uses `${PORT}`.
5. Deploy and wait for the Docker build to finish. The build needs outbound network access to download the dataset when `data/heart.csv` is not included.
6. When the service is live, test `https://YOUR-SERVICE.onrender.com/health`, `/info`, and `/docs`. Submit the actual Render URL, not the placeholder.

Example checks after replacing the placeholder:

```bash
https://heart-disease-fastapi-2x5d.onrender.com/docs#/Prediction/predict_predict_post
https://heart-disease-fastapi-2x5d.onrender.com/health
https://heart-disease-fastapi-2x5d.onrender.com/info
```

If the Docker build fails while fetching the data, download the dataset CSV from Kaggle, put it at `data/heart.csv`, commit it if permitted by the dataset's terms, and redeploy. Alternatively update `HEART_DATA_URL` in your deployment/build setup to a trusted accessible CSV URL. Never commit API keys or Kaggle credentials.
<img width="894" height="652" alt="image" src="https://github.com/user-attachments/assets/5b142a30-e613-4595-8d49-0507bc5b9fa5" />
<img width="1554" height="994" alt="image" src="https://github.com/user-attachments/assets/0e68eb49-def1-47cc-b39e-5decb0546d97" />



## Model and evaluation

The training script uses `StandardScaler` followed by `LogisticRegression(max_iter=2000)`. It creates a stratified 80/20 train/test split (`random_state=42`), writes `model/heart_model.joblib`, and records accuracy, ROC-AUC and the classification report in `model/metadata.json`. The actual metrics are printed and stored during training rather than guessed in this README.

