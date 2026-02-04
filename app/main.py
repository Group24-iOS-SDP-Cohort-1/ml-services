from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from app.clustering import run_hdbscan
from app.schemas import ClusterRequest, ClusterResponse

app = FastAPI(title="HDBSCAN Clustering Service")

# Load model ONCE at startup
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.post("/cluster", response_model=ClusterResponse)
def cluster_texts(payload: ClusterRequest):

    # ---------- Validation ----------
    if not payload.texts or len(payload.texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 text items are required for clustering"
        )

    if any(not t.strip() for t in payload.texts):
        raise HTTPException(
            status_code=400,
            detail="Texts must not be empty"
        )

    # ---------- Embeddings ----------
    try:
        embeddings = model.encode(
            payload.texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).tolist()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )

    # ---------- Clustering ----------
    return run_hdbscan(
        embeddings=embeddings,
        min_cluster_size=payload.min_cluster_size,
        min_samples=payload.min_samples
    )


@app.get("/health")
def health():
    return {"status": "ok"}


