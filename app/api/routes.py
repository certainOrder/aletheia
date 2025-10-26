"""Secondary API routes mounted under the `/api` prefix.

Currently provides a health check endpoint for liveness/readiness probes.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
