from locust import HttpUser, between, task


class RagApiUser(HttpUser):
    """轻量接口压测脚本；不会伪造吞吐量或延迟结果。"""

    wait_time = between(1, 3)

    @task(3)
    def chat(self):
        self.client.post("/api/chat", json={"query": "动力电池质保多久"}, name="POST /api/chat")

    @task(1)
    def retrieve(self):
        self.client.post("/api/retrieve", json={"query": "如何预约维修", "top_k": 3}, name="POST /api/retrieve")
