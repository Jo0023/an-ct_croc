from fastapi import FastAPI, HTTPException
from app.redis_client import get_cached_data, set_cached_data, check_redis_health
from app.qdrant_client import upsert_vector, search_vectors, check_qdrant_health
from app.models import Item, SearchResponse, CacheRequest, HealthResponse
import requests
import json
from datetime import datetime
from pydantic import BaseModel
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Search App - TP4 ML Integration",
    description="Application de recherche vectorielle avec intégration ML",
    version="2.0.0"
)

# Modèles pour l'intégration ML
class MLFeatures(BaseModel):
    features: list[float]

class MLPrediction(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict
    model_accuracy: float

class IntegratedHealthResponse(HealthResponse):
    ml_inference: str
    services: dict

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Vector Search Application - TP4 ML Integration",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "integrated_health": "/integrated-health",
            "ml_predict": "/ml-predict",
            "predict_example": "/predict-example",
            "add_item": "/items/",
            "search": "/search/",
            "cache": "/cache/{key}"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint original"""
    redis_status = check_redis_health()
    qdrant_status = check_qdrant_health()

    overall_status = "healthy"
    if redis_status == "error" and qdrant_status == "error":
        overall_status = "unhealthy"
    elif redis_status == "error" or qdrant_status == "error":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        redis=redis_status,
        qdrant=qdrant_status
    )

@app.get("/integrated-health", response_model=IntegratedHealthResponse, tags=["Health"])
def integrated_health_check():
    """Health check étendu avec service ML"""
    redis_status = check_redis_health()
    qdrant_status = check_qdrant_health()
    
    # Vérification du service ML
    try:
        ml_response = requests.get("http://inference-service/health", timeout=5)
        ml_status = "healthy"
        ml_details = ml_response.json()
    except Exception as e:
        ml_status = f"unhealthy: {str(e)}"
        ml_details = None

    overall_status = "healthy"
    error_count = sum(1 for status in [redis_status, qdrant_status] if status == "error")
    if error_count == 2:
        overall_status = "unhealthy"
    elif error_count == 1 or "unhealthy" in ml_status:
        overall_status = "degraded"

    return IntegratedHealthResponse(
        status=overall_status,
        redis=redis_status,
        qdrant=qdrant_status,
        ml_inference=ml_status,
        services={
            "redis": {"host": "redis-master", "port": 6379},
            "qdrant": {"host": "qdrant", "port": 6333},
            "ml_inference": {"host": "inference-service", "port": 80}
        }
    )

@app.post("/ml-predict", response_model=MLPrediction, tags=["ML Integration"])
def ml_predict(features: MLFeatures):
    """Prédiction ML intégrée avec cache Redis"""
    try:
        # Validation des features
        if len(features.features) != 4:
            raise HTTPException(
                status_code=422,
                detail="Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]"
            )
        
        # Vérifier le cache Redis
        cache_key = f"ml_pred:{hash(str(features.features))}"
        cached_result = get_cached_data(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for {cache_key}")
            return MLPrediction(**json.loads(cached_result))
        
        # Appel au service d'inférence
        ml_response = requests.post(
            "http://inference-service/predict",
            json={"features": features.features},
            timeout=10
        )
        
        if ml_response.status_code != 200:
            raise HTTPException(
                status_code=ml_response.status_code,
                detail=f"ML service error: {ml_response.text}"
            )
        
        ml_result = ml_response.json()
        
        # Cache du résultat (5 minutes)
        set_cached_data(cache_key, json.dumps(ml_result), expire=300)
        
        # Log de la prédiction
        log_entry = {
            "features": features.features,
            "prediction": ml_result.get('prediction'),
            "class_name": ml_result.get('class_name'),
            "timestamp": datetime.now().isoformat()
        }
        set_cached_data(f"ml_log:{datetime.now().timestamp()}", json.dumps(log_entry), expire=3600)
        
        return MLPrediction(**ml_result)
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="ML service timeout")
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/predict-example", tags=["ML Integration"])
def predict_example():
    """Exemples de prédictions ML"""
    examples = [
        {"features": [5.1, 3.5, 1.4, 0.2], "description": "Setosa - small petals"},
        {"features": [6.7, 3.0, 5.2, 2.3], "description": "Virginica - large petals"},
        {"features": [5.9, 2.8, 4.3, 1.3], "description": "Versicolor - medium petals"}
    ]
    
    results = []
    for example in examples:
        try:
            response = requests.post(
                "http://inference-service/predict",
                json={"features": example["features"]},
                timeout=5
            )
            results.append({
                "input": example,
                "output": response.json(),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "input": example,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "demo_predictions": results,
        "total_tested": len(results),
        "services_used": ["ml-inference", "redis-cache"],
        "integration": "successful"
    }

@app.get("/cache-stats", tags=["ML Integration"])
def cache_stats():
    """Statistiques du cache ML"""
    try:
        # Cette fonctionnalité nécessiterait une extension de redis_client
        # Pour la démo, retournons des stats basiques
        return {
            "cache_info": {
                "ml_predictions": "cached with 5min TTL",
                "prediction_logs": "stored for 1 hour",
                "integration": "active"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Garder les endpoints originaux
@app.post("/items/", tags=["Vectors"])
def add_item(item: Item):
    """Ajouter un élément avec vecteur à Qdrant"""
    try:
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=item.id,
            vector=item.vector,
            payload=item.payload or {},
        )
        result = upsert_vector(point)
        set_cached_data(f"item:{item.id}", "inserted")
        return {
            "status": "added",
            "item_id": item.id,
            "result": str(result)
        }
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding item: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding item: {str(e)}")

@app.get("/search/", response_model=list[SearchResponse], tags=["Vectors"])
def search_vectors_endpoint(vector: str, limit: int = 5):
    """Rechercher des vecteurs similaires"""
    try:
        vector_list = [float(x.strip()) for x in vector.split(",")]

        if len(vector_list) != 128:
            raise HTTPException(
                status_code=422,
                detail=f"Vector must have 128 dimensions, got {len(vector_list)}"
            )

        results = search_vectors(vector_list, limit)
        return [
            SearchResponse(
                id=hit.id,
                score=hit.score,
                payload=hit.payload,
            )
            for hit in results
        ]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid vector format: {str(e)}")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/cache/{key}", tags=["Cache"])
def get_cache(key: str):
    """Récupérer une valeur du cache Redis"""
    try:
        value = get_cached_data(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"key": key, "value": value}
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Cache error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.post("/cache/{key}", tags=["Cache"])
def set_cache(key: str, request: CacheRequest):
    """Stocker une valeur dans le cache Redis"""
    try:
        set_cached_data(key, request.value, request.expire)
        return {"status": "cached", "key": key, "expire_in": request.expire}
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Cache error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)