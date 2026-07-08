from fastapi import APIRouter

router = APIRouter(
    prefix="/v1",
    tags=["Models"],
)


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "MiniLlama",
                "object": "model",
                "created": 0,
                "owned_by": "MiniLlama",
            }
        ],
    }