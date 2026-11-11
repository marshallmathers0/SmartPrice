"""Day 5 — Repository Pattern."""
import pytest

from app.exceptions.pricing import PricingException, ProductNotFoundError
from app.repositories.base import ProductRepository
from app.repositories.mock import MockProductRepository


def test_repository_abstraction_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ProductRepository()


@pytest.mark.asyncio
@pytest.mark.parametrize("product_id", ["SKU-12345", "SKU-67890", "SKU-11111"])
async def test_mock_repository_serves_seed_catalog(product_id):
    product = await MockProductRepository().get_by_id(product_id)

    assert product.product_id == product_id


@pytest.mark.asyncio
async def test_mock_repository_accepts_a_custom_catalog(make_product):
    repository = MockProductRepository([make_product(product_id="TEST-1")])

    assert (await repository.get_by_id("TEST-1")).product_id == "TEST-1"
    with pytest.raises(ProductNotFoundError):
        await repository.get_by_id("SKU-12345")   # seed catalog is replaced, not merged


@pytest.mark.asyncio
async def test_missing_product_raises_product_not_found():
    with pytest.raises(ProductNotFoundError) as exc_info:
        await MockProductRepository().get_by_id("SKU-99999")

    error = exc_info.value
    assert isinstance(error, PricingException)
    assert error.product_id == "SKU-99999"
    assert error.code == "product_not_found"
    assert error.message == "Product 'SKU-99999' was not found"
