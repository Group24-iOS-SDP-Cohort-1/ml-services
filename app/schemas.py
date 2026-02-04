from pydantic import BaseModel
from typing import List, Optional, Dict

class ClusterRequest(BaseModel):
    texts: List[str]
    min_cluster_size: Optional[int] = 2
    min_samples: Optional[int] = 1

class ClusterResponse(BaseModel):
    labels: List[int]
    outliers: List[int]
    clusters: Dict[int, int]
    gaps: List[str]
