from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# -------- Request Schema --------
class ClusterRequest(BaseModel):
    texts: List[str]
    query: str
    min_cluster_size: Optional[int] = 2
    min_samples: Optional[int] = 1
