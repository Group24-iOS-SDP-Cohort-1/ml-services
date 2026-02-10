import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def extract_json(text: str):
    text = text.strip()

    # 1. Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # 2. Extract first JSON-like block
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


def analyze_clusters_with_gemini(payload: dict):

    prompt = f"""
You are an elite YouTube Shorts trend analyst.

User query: "{payload['query']}"

For EACH cluster, generate exactly ONE scroll-stopping Shorts idea.
Your job is to find CONTENT GAPS creators are NOT doing yet.

STRICT RULES:
- Output must be VALID JSON ONLY (no markdown, no extra text)
- Exactly 1 idea per cluster
- Title: max 12 words, catchy + YouTube-attractive
- Description: ONE sentence, max 15 words
- Format: one of [asmr, challenge, advice, tutorial, storytime, aesthetic, comedy]
- Hashtags: EXACTLY 5, each must start with "#"
- No multiline text, no unfinished quotes, no trailing commas

DO NOT suggest:
- testing viral hacks
- reviewing weird tools
- generic transformations
- reaction videos

Each idea must target a clear audience (teen girls, Indian creators, beginners, etc.)

Clusters:
{json.dumps(payload['clusters'], indent=2)}

Return ONLY valid JSON in this exact structure:

{{
  "cluster_analysis": [
    {{
      "cluster_id": 0,
      "theme": "...",
      "gaps": ["..."],
      "ideas": [
        {{
          "title": "...",
          "description": "...",
          "format": "...",
          "hashtags": ["#tag1","#tag2","#tag3","#tag4","#tag5"],
          "noveltyScore": 1-10
        }}
      ]
    }}
  ]
}}
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "max_output_tokens": 4096   # safer, avoids truncation
        }
    )
    finish_reason = response.candidates[0].finish_reason

    print(finish_reason)

    raw_text = response.text.strip()

    print("🔥 RAW GEMINI OUTPUT:\n", raw_text)
    
    if finish_reason == "MAX_TOKENS":
        print("⚠ Retrying Gemini due to truncation")

        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 3072
            }
        )

        raw_text = response.text.strip()


    if not raw_text:
        return {
            "cluster_analysis": [],
            "raw_text": "",
            "error": "Gemini returned empty response"
        }

    # Try parsing JSON
    parsed = extract_json(raw_text)

    # Case 2: JSON parsed successfully
    if parsed is not None:
        return {
            "cluster_analysis": parsed.get("cluster_analysis", []),
            "raw_text": raw_text,
            "success": True
        }

    # Case 3: JSON invalid → send raw text fallback
    return {
        "cluster_analysis": [],
        "raw_text": raw_text,
        "error": "Gemini output malformed, returning raw text"
    }
