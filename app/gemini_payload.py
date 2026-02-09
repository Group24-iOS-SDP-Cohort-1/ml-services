def sanitize_text(text: str):
    """
    Clean dangerous characters that break Gemini JSON output.
    Removes quotes, newlines, weird unicode.
    """
    if not text:
        return ""

    return (
        text.replace('"', '')          # remove quotes
            .replace("\n", " ")        # remove newlines
            .replace("\r", " ")
            .replace("’", "'")         # normalize apostrophes
            .strip()
    )


def build_gemini_payload(raw_result: dict, query: str):

    unified_labels = raw_result.get("unified_labels", {})
    videos = raw_result.get("videos", [])

    cluster_list = []

    # -------------------------------
    # Extract Cluster Metadata Properly
    # -------------------------------
    for cid, obj in unified_labels.items():

        # ✅ Clean examples safely (ONLY 2)
        examples = [
            sanitize_text(e)
            for e in obj.get("examples", [])[:2]
            if e
        ]

        # ✅ Clean keywords too
        keywords = [
            sanitize_text(k)
            for k in obj.get("keywords", [])[:6]
        ]

        cluster_list.append({
            "cluster_id": int(cid),
            "title": sanitize_text(obj.get("title", "")),
            "keywords": keywords,
            "examples": examples,
            "description": sanitize_text(obj.get("description", ""))
        })

    # ✅ Limit clusters sent to Gemini (Top 5 only)
    cluster_list = cluster_list[:5]

    # -------------------------------
    # Sample Outliers (Clean them too)
    # -------------------------------
    outliers = [
        sanitize_text(v["title"])
        for v in videos
        if v.get("cluster") == -1
    ][:3]   # only 3 outliers

    return {
        "query": sanitize_text(query),
        "clusters": cluster_list,
        "outliers_sample": outliers
    }
