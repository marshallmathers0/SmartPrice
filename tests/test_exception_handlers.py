"""Day 5 — Global exception handlers."""
import pytest
from fastapi.testclient import TestClient

from app.exceptions.pricing import PricingException


class OutOfStockError(PricingException):
    """A business exception with no entry in the status-code table."""

    code = "out_of_stock"


@pytest.fixture
def error_client(app):
    """Test app with two extra routes that raise on purpose."""

    @app.get("/test/business-error")
    async def business_error():
        raise OutOfStockError("Product 'SKU-12345' is out of stock")

    @app.get("/test/crash")
    async def crash():
        raise RuntimeError("connection failed: postgresql://admin:secret@db/smartprice")

    # raise_server_exceptions=False: inspect the 500 response instead of re-raising in the test
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_unmapped_business_exception_falls_back_to_400(error_client):
    response = error_client.get("/test/business-error")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Product 'SKU-12345' is out of stock",
        "code": "out_of_stock",
    }


def test_unexpected_exception_returns_generic_500_without_leaking_details(error_client):
    response = error_client.get("/test/crash")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "code": "internal_error"}
    assert "secret" not in response.text
