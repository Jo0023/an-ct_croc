from fastapi import FastAPI, HTTPException
from app.redis_client import get_cached_data, set_cached_data, check_redis_health
from app.qdrant_client import upsert_vector, search_vectors, check_qdrant_health
from app.models import Item, SearchResponse, CacheRequest, HealthResponse
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Search App",
    description="Application de recherche vectorielle avec Redis et Qdrant",
    version="1.0.0"
)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Vector Search Application",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "add_item": "/items/",
            "search": "/search/",
            "cache": "/cache/{key}"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint pour Kubernetes"""
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
        # Convertir la string en liste de floats
        vector_list = [float(x.strip()) for x in vector.split(",")]

        # Valider la taille du vecteur
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