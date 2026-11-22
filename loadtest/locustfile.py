"""Load test: `uv run locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 30s --host http://localhost:8000`.

Run the app with config/routes.loadtest.json (see README). Set LOCUST_API_KEY to a key from `python -m gateway.cli create-key` (use a high --rpm).
"""

import os

from locust import HttpUser, between, task

KEY = os.environ.get("LOCUST_API_KEY", "")
H = {"Authorization": f"Bearer {KEY}"}


class ChatUser(HttpUser):
    wait_time = between(0, 0.05)

    @task(8)
    def chat(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={"model": "smart", "messages": [{"role": "user", "content": "ping"}]},
            headers=H,
            name="chat",
        )

    @task(1)
    def stream(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "smart",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers=H,
            name="chat (stream)",
        )
