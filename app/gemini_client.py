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

    # Try full direct parse first
    try:
        return json.loads(text)
    except:
        pass

    # Match object OR array
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)

    if not match:
        print("❌ No JSON found in output")
        print("🔥 RAW OUTPUT:\n", text)
        return None

    try:
        return json.loads(match.group())
    except Exception as e:
        print("❌ JSON parse failed:", e)
        print("🔥 RAW OUTPUT:\n", text)
        return None


def analyze_clusters_with_gemini(payload: dict):

    prompt = f"""
You are an elite YouTube Shorts trend analyst.

User query: "{payload['query']}"

You are a YouTube trend idea generator.

I will give you clusters of trending video themes.
For EACH cluster, generate exactly ONE scroll-stopping video idea.

Your job is to find CONTENT GAPS creators are NOT doing yet.

STRICT RULES:
- Output must be VALID JSON ONLY (no markdown, no commentary)
- Every JSON value must be a SINGLE LINE string
- No multiline text, no unfinished quotes
- Exactly 1 idea per cluster
- Do NOT suggest:
  • testing viral hacks
  • reviewing weird tools
  • generic transformations
  • reaction videos
- Title must be catchy, fun, and YouTube-attractive
- Hook must be <= 8 words (spoken in 1 second)
- Each idea must include ONE unexpected twist
- Each idea must target a clear audience (teen girls, Indian creators, beginners, etc.)

Clusters:
{json.dumps(payload['clusters'], indent=2)}

Return ONLY valid JSON:

{{
  "cluster_analysis": [
    {{
      "cluster_id": 0,
      "theme": "...",
      "gaps": [
        "Underserved audience gap",
        "Missing format gap"
      ],
      "ideas": [
        {{
          "title": "Scroll-stopping Shorts title",
          "hook": "<=8 word viral hook",
          "format": "POV / Challenge / Split-screen / AI filter / Storytime",
          "whyItWillTrend": "Specific reason this gap will explode",
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
            "max_output_tokens": 4096
        }
    )

    print("🔥 RAW GEMINI OUTPUT:\n", response.text)

    return json.loads(response.text)
