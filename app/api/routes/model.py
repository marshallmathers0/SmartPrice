import logging

from fastapi import APIRouter, status

from app.api.dependencies import ModelBundleDep
from app.schemas.errors import ErrorResponse
from app.schemas.model import ModelInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Model"])


@router.get(
    "/current",
    response_model=ModelInfoResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "No model loaded"}},
)
async def current_model(bundle: ModelBundleDep) -> ModelInfoResponse:
    """
    GET /api/v1/models/current

    Which model version is serving, what it was trained on (dataset hash),
    and how well it did on validation and on the held-out test set.
    """
    return ModelInfoResponse.from_metadata(bundle.metadata)
