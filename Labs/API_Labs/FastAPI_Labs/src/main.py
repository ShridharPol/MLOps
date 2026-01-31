from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import joblib

from .predict import predict_data

app = FastAPI()


# ---------- Request / Response Schemas ----------

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float


class IrisResponse(BaseModel):
    response: int
    predicted_class_name: str
    probabilities: Dict[str, float]


# ---------- Health Check ----------

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


# ---------- Predict Endpoint ----------

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]]

        result = predict_data(features)  # returns dict
        return IrisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Model Info Endpoint (Option B) ----------

@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Returns metadata about the model and expected feature order.
    This does NOT affect /predict.
    """

    # Must match the order used in /predict
    features_order: List[str] = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    class_mapping: Dict[int, str] = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }

    try:
        # IMPORTANT: keep the same relative path style you’re using elsewhere
        model = joblib.load("./model/iris_model.pkl")

        # Some sklearn models may include numpy types; keep it JSON-friendly
        params = model.get_params() if hasattr(model, "get_params") else {}

        return {
            "model_type": type(model).__name__,
            "model_params": params,
            "features_order": features_order,
            "class_mapping": class_mapping,
            "supports_predict_proba": hasattr(model, "predict_proba"),
        }

    except Exception as e:
        # Fallback: still returns useful info instead of crashing
        return {
            "model_type": "unknown (load failed)",
            "error": str(e),
            "features_order": features_order,
            "class_mapping": class_mapping,
            "supports_predict_proba": True,
        }
