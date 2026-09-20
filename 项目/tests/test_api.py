def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_retrieve_api(client):
    response = client.post("/api/retrieve", json={"query": "动力电池质保多久", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "动力电池质保多久"
    assert len(body["documents"]) == 3
    assert "document_id" in body["documents"][0]


def test_chat_api(client):
    response = client.post("/api/chat", json={"query": "怎么预约维修"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["retrieved_document_ids"]
    assert body["latency_ms"] >= 0
