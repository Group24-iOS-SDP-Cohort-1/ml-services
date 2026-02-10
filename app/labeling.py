from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

kw_model = KeyBERT(embedding_model)
# -------- Unified Cluster Summary --------
def generate_cluster_summary(keywords):
    if not keywords:
        keywords = ["general"]

    title = " / ".join(keywords[:3]).title()

    description = f"Trending videos related to {', '.join(keywords[:4])}."

    return title, description

# =====================================================
# Keyword Extractor (KeyBERT)
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
