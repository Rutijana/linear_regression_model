"""Farm Price Predictor API.

Predicts the expected market price (RWF per unit) of an agricultural commodity in a
Rwandan market for a given month and price type, using the model trained in
linear_regression/multivariate.ipynb.

Run locally:  uvicorn prediction:app --reload   (from summative/API)
"""

import gc
import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

API_DIR = Path(__file__).resolve().parent
MODEL_PATH = API_DIR / "best_model.pkl"
METADATA_PATH = API_DIR / "model_metadata.json"
TRAINING_DATA_PATH = API_DIR / "training_data.csv"
UPLOADED_ROWS_PATH = API_DIR / "uploaded_rows.csv"  # rows added through /retrain (cumulative)

# Same split as the notebook, so the retrain MSE is measured on the same test rows
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
YEAR_MAX_ALLOWED = 2030

# ---------------------------------------------------------------- load model + metadata at startup
with open(METADATA_PATH, encoding="utf-8") as f:
    METADATA = json.load(f)

NUMERIC_FEATURES = METADATA["model_features"]["numeric"]
CATEGORICAL_FEATURES = METADATA["model_features"]["categorical"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = METADATA["target"]
MARKET_LOOKUP = METADATA["lookups"]["market"]
COMMODITY_LOOKUP = METADATA["lookups"]["commodity"]
ALLOWED = METADATA["allowed_values"]
YEAR_MIN = METADATA["numeric_ranges"]["year"]["min"]
YEAR_MAX_DATA = METADATA["numeric_ranges"]["year"]["max"]

model = joblib.load(MODEL_PATH)
model_info = {"name": METADATA["best_model"], "test_mse": METADATA["test_metrics"]["test_MSE"],
              "retrained_at": None}
model_lock = threading.Lock()   # guards reading/swapping the in-memory model
status_lock = threading.Lock()  # guards retrain_status (one retrain job at a time)
retrain_status = {
    "state": "idle",  # idle | running | done | failed
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "new_rows": None,
    "rows_used_for_training": None,
    "old_test_mse": None,
    "new_test_mse": None,
    "new_test_rmse_rwf": None,
    "training_seconds": None,
    "error": None,
}

# Literal types built from the training data, so Swagger shows the allowed values as a dropdown
CommodityName = Literal[tuple(ALLOWED["commodity"])]
MarketName = Literal[tuple(ALLOWED["market"])]
PriceType = Literal[tuple(ALLOWED["pricetype"])]

app = FastAPI(
    title="Farm Price Predictor API",
    version="1.0.0",
    description=(
        "Predicts the expected market price (RWF per unit) of an agricultural commodity in a Rwandan "
        "market for a given month and price type, so smallholder farmers can decide where and when to "
        "sell and buyers can make fair offers.\n\n"
        f"Model: **{METADATA['best_model']}** trained on WFP Rwanda food prices ({YEAR_MIN}-{YEAR_MAX_DATA}).\n\n"
        "**Wholesale caveat:** wholesale data covers only 3 commodities in 2 markets, so wholesale "
        "predictions are less reliable than retail ones."
    ),
)

# CORS (Cross-Origin Resource Sharing) decides which *websites* may call this API from a browser.
# - allow_origins: only our own deployed site and local Flutter-web debug servers. No "*", so a random
#   website can't use this API from its visitors' browsers. Replace the Render placeholder with the
#   real URL after deploying. (Run Flutter web with `flutter run -d chrome --web-port 8080` to match.)
# - allow_methods: only GET (health/docs), POST (/predict, /retrain) and OPTIONS (browser preflight).
#   PUT/PATCH/DELETE are not used, so they are blocked.
# - allow_headers: only Content-Type (JSON body / multipart upload) and Accept are needed.
# - allow_credentials=False: the API uses no cookies or auth headers, so browsers must not send
#   credentials. This keeps the attack surface small.
# Note: the Flutter *mobile* app is not a browser, so CORS does not restrict it. CORS only controls
# which web pages (origins) can call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://farm-price-predictor.onrender.com",  # placeholder: replace with the real Render URL
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


class PriceRequest(BaseModel):
    """One prediction request. Location and commodity details are looked up from market/commodity."""

    # strict: "2025" (string) is a wrong type for year, not silently converted; extra fields are rejected
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"example": {"commodity": "Maize", "market": "Kimironko",
                                       "pricetype": "Retail", "year": 2025, "month": 3}},
    )

    commodity: CommodityName = Field(
        description=f"Commodity to price ({len(ALLOWED['commodity'])} allowed values), e.g. Maize, "
                    "Beans (dry), Potatoes (Irish). Its unit (KG, L, Unit, Sack) is returned with the price.",
        examples=["Maize"],
    )
    market: MarketName = Field(
        description=f"Rwandan market where the commodity is sold ({len(ALLOWED['market'])} allowed values), "
                    "e.g. Kimironko, Nyabugogo, Musanze. Its district, province and coordinates are filled in "
                    "automatically.",
        examples=["Kimironko"],
    )
    pricetype: PriceType = Field(
        description="Retail or Wholesale. Wholesale data covers only 3 commodities in 2 markets, so "
                    "wholesale predictions are less reliable.",
        examples=["Retail"],
    )
    year: int = Field(
        ge=YEAR_MIN, le=YEAR_MAX_ALLOWED,
        description=f"Year of sale ({YEAR_MIN}-{YEAR_MAX_ALLOWED}). The data ends in {YEAR_MAX_DATA}: "
                    f"years after {YEAR_MAX_DATA} are treated like {YEAR_MAX_DATA}, the latest year the "
                    "model has seen (no inflation is extrapolated).",
        examples=[2025],
    )
    month: int = Field(ge=1, le=12, description="Month of sale (1 = January ... 12 = December). "
                                                "Captures harvest seasonality.", examples=[3])


class PriceResponse(BaseModel):
    predicted_price_rwf: float = Field(description="Expected price in Rwandan francs per unit")
    unit: str = Field(description="Unit the price refers to (KG, L, Unit or Sack)")
    commodity: str
    market: str
    pricetype: str
    year: int
    month: int


class RetrainStarted(BaseModel):
    message: str
    job_id: str
    new_rows: int
    status_url: str


class RetrainStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    job_id: str | None
    started_at: str | None
    finished_at: str | None
    new_rows: int | None
    rows_used_for_training: int | None
    old_test_mse: float | None
    new_test_mse: float | None
    new_test_rmse_rwf: float | None
    training_seconds: float | None
    error: str | None


def build_features(req: PriceRequest) -> pd.DataFrame:
    """Turn the 5 farmer inputs into the 11 model features using the saved lookup tables."""
    row = {
        "commodity": req.commodity,
        "market": req.market,
        "pricetype": req.pricetype,
        "year": req.year,
        "month": req.month,
        **MARKET_LOOKUP[req.market],        # latitude, longitude, admin1, admin2
        **COMMODITY_LOOKUP[req.commodity],  # category, unit
    }
    return pd.DataFrame([row])[FEATURES]


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "message": "Farm Price Predictor API. Open /docs to test it (Swagger UI).",
        "docs": "/docs",
        "model": model_info["name"],
        "retrained_at": model_info["retrained_at"],
    }


@app.post("/predict", response_model=PriceResponse, tags=["prediction"])
def predict(req: PriceRequest):
    features = build_features(req)
    with model_lock:
        current_model = model  # the old model keeps serving until a retrain swaps in the new one
    price = float(current_model.predict(features)[0])
    return PriceResponse(
        predicted_price_rwf=round(max(price, 0.0), 2),
        unit=COMMODITY_LOOKUP[req.commodity]["unit"],
        **req.model_dump(),
    )


def validate_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, types, ranges and categorical values of an uploaded retraining CSV (400 on error)."""
    expected = FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise HTTPException(status_code=400, detail={
            "error": "CSV columns do not match the training data",
            "expected_columns": expected, "missing_columns": missing, "unexpected_columns": extra,
        })
    df = df[expected].copy()
    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has a header but no data rows.")
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise HTTPException(status_code=400, detail=f"Missing values in columns: {bad}")

    for col in NUMERIC_FEATURES + [TARGET]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise HTTPException(status_code=400, detail=f"Column '{col}' must contain only numbers.")
        df[col] = converted
    for col in ["year", "month"]:
        if (df[col] % 1 != 0).any():
            raise HTTPException(status_code=400, detail=f"Column '{col}' must contain whole numbers.")
        df[col] = df[col].astype(int)

    errors = []
    if not df["year"].between(YEAR_MIN, YEAR_MAX_ALLOWED).all():
        errors.append(f"year must be between {YEAR_MIN} and {YEAR_MAX_ALLOWED}")
    if not df["month"].between(1, 12).all():
        errors.append("month must be between 1 and 12")
    if not (df[TARGET] > 0).all():
        errors.append("price must be greater than 0")
    for col in ["commodity", "market", "pricetype"]:
        unknown = sorted(set(df[col].astype(str)) - set(ALLOWED[col]))
        if unknown:
            errors.append(f"unknown {col} values: {unknown[:10]}")
    if errors:
        raise HTTPException(status_code=400, detail={"error": "Invalid values in CSV", "problems": errors})

    # Derived columns must match the lookups, then use the canonical values
    for col, lookup, key in [("latitude", MARKET_LOOKUP, "market"), ("longitude", MARKET_LOOKUP, "market"),
                             ("admin1", MARKET_LOOKUP, "market"), ("admin2", MARKET_LOOKUP, "market"),
                             ("category", COMMODITY_LOOKUP, "commodity"), ("unit", COMMODITY_LOOKUP, "commodity")]:
        df[col] = df[key].map(lambda k: lookup[k][col])
    return df


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_retrain(job_id: str, new_rows: pd.DataFrame) -> None:
    """Background job: refit the model on the original training rows + all uploaded rows, then swap it in.

    The best model is a Random Forest (not SGD), so it can't continue training with partial_fit: it is
    refitted from scratch and evaluated on the same held-out test rows as in the notebook.
    """
    global model
    try:
        start = time.perf_counter()
        base = pd.read_csv(TRAINING_DATA_PATH)
        base_train, base_test = train_test_split(base, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        del base

        previous = pd.read_csv(UPLOADED_ROWS_PATH) if UPLOADED_ROWS_PATH.exists() else None
        all_uploaded = pd.concat([previous, new_rows], ignore_index=True) if previous is not None else new_rows
        train = pd.concat([base_train, all_uploaded], ignore_index=True)
        del base_train

        # Same pipeline and hyperparameters as the saved model; one core keeps memory low on Render
        with model_lock:
            new_model = clone(model)
        new_model.set_params(model__regressor__n_jobs=1)
        new_model.fit(train[FEATURES], train[TARGET])
        rows_used = len(train)
        del train
        gc.collect()

        preds = np.clip(new_model.predict(base_test[FEATURES]), 0, None)
        mse = float(mean_squared_error(base_test[TARGET], preds))

        # Persist first (atomic file replace), then swap the in-memory model under the lock
        tmp_path = MODEL_PATH.with_suffix(".tmp")
        joblib.dump(new_model, tmp_path, compress=3)
        tmp_path.replace(MODEL_PATH)
        all_uploaded.to_csv(UPLOADED_ROWS_PATH, index=False)
        with model_lock:
            old_mse = model_info["test_mse"]
            model = new_model
            model_info.update(test_mse=mse, retrained_at=now())
        elapsed = time.perf_counter() - start

        with status_lock:
            retrain_status.update(
                state="done", finished_at=now(), rows_used_for_training=rows_used,
                old_test_mse=round(old_mse, 2), new_test_mse=round(mse, 2),
                new_test_rmse_rwf=round(mse ** 0.5, 2), training_seconds=round(elapsed, 1),
            )
    except Exception as exc:  # report the failure through /retrain/status; the old model keeps serving
        with status_lock:
            retrain_status.update(state="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()


@app.post("/retrain", status_code=202, response_model=RetrainStarted, tags=["retraining"],
          responses={400: {"description": "Invalid CSV"}, 409: {"description": "A retraining is already running"}})
def retrain(file: UploadFile = File(description="CSV with the same columns as training_data.csv: "
                                                 + ", ".join(FEATURES + [TARGET]))):
    """Upload new price rows. The CSV is validated, then retraining starts in the background right away.

    Returns 202 with a job id; follow progress at GET /retrain/status. /predict keeps using the current
    model until the new one is ready, then switches to it automatically (no redeploy).
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")
    try:
        new_rows = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as CSV.")
    new_rows = validate_upload(new_rows)

    with status_lock:
        if retrain_status["state"] == "running":
            raise HTTPException(status_code=409, detail={
                "error": "A retraining is already running. Try again when it has finished.",
                "job_id": retrain_status["job_id"], "status_url": "/retrain/status"})
        job_id = uuid.uuid4().hex[:12]
        retrain_status.update(
            state="running", job_id=job_id, started_at=now(), finished_at=None, new_rows=len(new_rows),
            rows_used_for_training=None, old_test_mse=None, new_test_mse=None, new_test_rmse_rwf=None,
            training_seconds=None, error=None,
        )
    threading.Thread(target=run_retrain, args=(job_id, new_rows), daemon=True).start()

    return RetrainStarted(message="Training started. Check GET /retrain/status for the result.",
                          job_id=job_id, new_rows=len(new_rows), status_url="/retrain/status")


@app.get("/retrain/status", response_model=RetrainStatus, tags=["retraining"])
def retrain_job_status():
    """State of the latest retraining job: idle, running, done or failed, with old vs new test MSE."""
    with status_lock:
        return RetrainStatus(**retrain_status)
