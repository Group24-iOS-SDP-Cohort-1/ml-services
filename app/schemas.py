from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# -------- Request Schema --------
class ClusterRequest(BaseModel):
    texts: List[str]
    query: str
    min_cluster_size: Optional[int] = 2
    min_samples: Optional[int] = 1


# -------- Cluster Label Schema --------
class ClusterLabel(BaseModel):
    title: str
    description: str
    keywords: List[str]
    examples: List[str]


# -------- Response Schema --------
class GeminiIdea(BaseModel):
    title: str
    hook: str
    format: str
    whyItWillTrend: str


class GeminiClusterAnalysis(BaseModel):
    cluster_id: int
    theme: str
    gaps: List[str]
    ideas: List[GeminiIdea]


class ClusterResponse(BaseModel):
    labels: List[int]
    outliers: List[int]

    # ✅ cluster_id -> size
    clusters: Dict[str, int]

    # ✅ cluster_id -> list of indices
    cluster_texts: Dict[str, List[int]]

    # ✅ cluster_id -> label object
    unified_labels: Dict[str, ClusterLabel]

    gaps: List[str]

    gemini_analysis: Optional[List[GeminiClusterAnalysis]] = None

