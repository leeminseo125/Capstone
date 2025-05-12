from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os

app = FastAPI()

class RegionInfo(BaseModel):
    region_id: int
    head_count: int
    congestion: str

class CrowdInfo(BaseModel):
    timestamp: str
    regions: List[RegionInfo]

def calc_congestion(head_count: int) -> str:
    if head_count < 5:
        return "여유"
    elif head_count < 15:
        return "보통"
    else:
        return "혼잡"

@app.get("/crowd_info", response_model=CrowdInfo)
async def get_crowd_info():
    filepath = "shared_data.json"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=503, detail="추론 결과가 아직 없습니다.")

    with open(filepath, 'r') as f:
        data = json.load(f)

    regions = [
        RegionInfo(
            region_id=i+1,
            head_count=count,
            congestion=calc_congestion(count)
        ) for i, count in enumerate(data["counts"])
    ]

    return CrowdInfo(
        timestamp=data["timestamp"],
        regions=regions
    )
