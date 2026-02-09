from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from dotenv import load_dotenv
load_dotenv()

from app.schemas import ClusterRequest, ClusterResponse
from app.clustering import run_hdbscan

#  Gemini Imports
from app.gemini_client import analyze_clusters_with_gemini
from app.gemini_payload import build_gemini_payload


app = FastAPI(title="HDBSCAN Clustering + Labelling Service")

# =====================================================
#  Load Models Once
# =====================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# KeyBERT uses the same embedding model
kw_model = KeyBERT(embedding_model)


# =====================================================
#  Example Cleaner
# =====================================================

def clean_example(text: str, max_len=140):
    """
    Shortens long keyword-stuffed examples.
    Keeps first sentence only.
    Removes spam like telegram/link promos.
    """
    if not text:
        return ""

    blacklist = ["telegram", "join group", "link in bio", "follow for link"]
    lowered = text.lower()

    if any(bad in lowered for bad in blacklist):
        return ""

    first_sentence = text.split(".")[0]
    return first_sentence[:max_len].strip()


# =====================================================
# ✅ Keyword Extractor (KeyBERT)
# =====================================================

def extract_keywords(examples: list, top_n=6):
    """
    Extract meaningful cluster keywords using KeyBERT.
    Uses cluster examples as context.
    """
    if not examples:
        return []

    joined_text = " ".join(examples)

    keywords = kw_model.extract_keywords(
        joined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )

    return [kw[0] for kw in keywords]


# =====================================================
# ✅ Compress unified_labels Safely
# =====================================================

def compress_unified_labels(unified_labels: dict):
    """
    Compress unified_labels while keeping schema unchanged.
    Adds KeyBERT semantic keywords.
    """

    for cid, label_obj in unified_labels.items():

        # Convert Pydantic → dict safely
        if hasattr(label_obj, "dict"):
            label_obj = label_obj.dict()

        # ---- Clean Examples ----
        examples = label_obj.get("examples", [])[:6]
        cleaned = [clean_example(e) for e in examples]
        cleaned = [e for e in cleaned if e]

        # Keep only top 3 examples
        label_obj["examples"] = cleaned[:3]

        # ---- Replace Keywords with KeyBERT ----
        label_obj["keywords"] = extract_keywords(cleaned)

        unified_labels[cid] = label_obj

    return unified_labels


# =====================================================
# ✅ Main Endpoint
# =====================================================

@app.post("/cluster")
def cluster_texts(payload: ClusterRequest):

    # -------- Validation --------
    if not payload.texts or len(payload.texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 text items are required for clustering"
        )
    
    if not payload.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query is required"
        )

    if any(not t.strip() for t in payload.texts):
        raise HTTPException(
            status_code=400,
            detail="Texts must not be empty"
        )

    # -------- Embeddings --------
    try:
        embeddings = embedding_model.encode(
            payload.texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).tolist()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )

    # -------- Clustering + Labelling --------
    raw_result = run_hdbscan(
        embeddings=embeddings,
        texts=payload.texts,
        min_cluster_size=payload.min_cluster_size,
        min_samples=payload.min_samples
    )

    # =====================================================
    # ✅ Compress Cluster Labels (KeyBERT)
    # =====================================================
    if "unified_labels" in raw_result:
        raw_result["unified_labels"] = compress_unified_labels(
            raw_result["unified_labels"]
        )

    # =====================================================
    # ✅ Gemini Cluster Gap + Idea Analysis
    # =====================================================
    try:
        gemini_payload = build_gemini_payload(
            raw_result,
            query=payload.query   # replace with payload.query if you add it
        )

        print("🔥 Payload sent to Gemini:", gemini_payload)

        gemini_analysis = analyze_clusters_with_gemini(gemini_payload)

        print("🔥 Gemini returned:", gemini_analysis)


        # Attach Gemini output into response
        raw_result["gemini_analysis"] = gemini_analysis

    except Exception as e:
        print("⚠️ Gemini failed:", str(e))
        raw_result["gemini_analysis"] = None


    return raw_result


# =====================================================
# ✅ Health Check
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}
