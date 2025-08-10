from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def read_root():
    return {"message": "Hello World"}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}