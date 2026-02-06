from sklearn.feature_extraction.text import TfidfVectorizer


# -------- Keyword Extraction --------
def extract_keywords(texts, top_k=6):
    """
    Extract top keywords from a cluster using TF-IDF.
    Handles empty clusters and stopword-only documents safely.
    """

    # 1. Clean input texts
    texts = [t.strip() for t in texts if t and t.strip()]

    # 2. If nothing remains, return fallback
    if len(texts) == 0:
        return ["general"]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    try:
        X = tfidf.fit_transform(texts)
    except ValueError:
        # Happens when all words are removed as stopwords
        return ["general"]

    scores = X.sum(axis=0).A1
    words = tfidf.get_feature_names_out()

    if len(words) == 0:
        return ["general"]

    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

    return [w for w, _ in ranked[:top_k]]

# -------- Unified Cluster Summary --------
def generate_cluster_summary(keywords):
    if not keywords:
        keywords = ["general"]

    title = " / ".join(keywords[:3]).title()

    description = f"Trending videos related to {', '.join(keywords[:4])}."

    return title, description
