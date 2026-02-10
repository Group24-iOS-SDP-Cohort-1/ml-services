def sanitize_text(text: str):
    if not text:
        return ""

    return (
        text.replace('"', '')
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("’", "'")
            .strip()
    )


def build_gemini_payload(raw_result: dict, query: str):

    unified_labels = raw_result.get("unified_labels", {})

    cluster_list = []

    for cid, obj in unified_labels.items():

        # -------------------------
        # PRIMARY SIGNAL → DESCRIPTION
        # -------------------------
        desc = sanitize_text(obj.get("description", ""))

        # Fallback → if description missing
        if not desc:
            desc = " ".join(obj.get("keywords", [])[:3])

        # Remove boilerplate phrase
        desc = desc.replace("Trending videos related to", "")

        # HARD TOKEN SAFETY LIMIT
        desc = desc[:120]

        cluster_list.append({
            "cluster_id": int(cid),
            "description": desc
        })

    return {
        "query": sanitize_text(query),
        "clusters": cluster_list[:3],   # VERY IMPORTANT → token safety
        "outliers_sample": []           # Not needed for Gemini
    }



