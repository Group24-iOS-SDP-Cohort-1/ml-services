def clean_example(text: str, max_len=140):
    """
    Cleans long keyword-stuffed examples.
    """
    if not text:
        return ""

    # remove spammy link phrases
    blacklist = ["telegram", "link in bio", "join group"]
    lowered = text.lower()

    if any(b in lowered for b in blacklist):
        return ""

    # keep first sentence only
    first = text.split(".")[0]

    return first[:max_len].strip()


def compress_clusters(raw_clusters: dict):
    compressed = []

    for cid, cluster in raw_clusters.items():

        title = cluster.get("title", "")
        keywords = cluster.get("keywords", [])[:8]

        examples = cluster.get("examples", [])[:3]
        cleaned = [clean_example(e) for e in examples]
        cleaned = [e for e in cleaned if e]  # remove empty ones

        compressed.append({
            "cluster_id": int(cid),
            "title": title,
            "keywords": keywords,
            "top_examples": cleaned,
            "video_count": len(cluster.get("examples", []))
        })

    return compressed
