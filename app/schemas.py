from pydantic import BaseModel
from typing import List, Optional, Dict


# -------- Request Schema --------
class ClusterRequest(BaseModel):
    texts: List[str]
    min_cluster_size: Optional[int] = 2
    min_samples: Optional[int] = 1


# -------- Cluster Label Schema --------
class ClusterLabel(BaseModel):
    title: str
    description: str
    keywords: List[str]
    examples: List[str]


# -------- Response Schema --------
class ClusterResponse(BaseModel):
    labels: List[int]
    outliers: List[int]

    clusters: Dict[int, int]                 # cluster_id → size
    cluster_texts: Dict[int, List[int]]      # cluster_id → indices

    unified_labels: Dict[int, ClusterLabel]  # cluster_id → label object

    gaps: List[str]
