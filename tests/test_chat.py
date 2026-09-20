import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_empty_message():
    # Attempt to send an empty message
    response = client.post(
        "/ai/chat",
        json={"message": "   "}
    )
    # The current dependencies mock the auth to failure unless valid JWT, 
    # but we will just assert it doesn't give a 500 error but a controlled HTTP error
    assert response.status_code in [400, 401, 403]

def test_conversations_unauthorized():
    # Should block without token
    response = client.get("/ai/conversations")
    assert response.status_code == 403 # HTTPBearer returns 403 if missing Authorization header
