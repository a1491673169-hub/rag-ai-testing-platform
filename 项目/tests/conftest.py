import pytest
from fastapi.testclient import TestClient

from app.main import app, service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def rag_service():
    return service
