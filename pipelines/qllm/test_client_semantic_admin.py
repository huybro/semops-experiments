from client_semantic_admin import run_checks


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def request(self, method, url, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_run_checks_hits_all_endpoints_in_order():
    session = FakeSession(
        [
            FakeResponse(200, {"budget": 123}),
            FakeResponse(200, {"status": "ok"}),
            FakeResponse(200, {"count": 1, "pinned_requests": [{"request_id": "r1"}]}),
            FakeResponse(200, {"queue_depth": 0}),
            FakeResponse(
                200,
                {
                    "unpinned_count": 1,
                    "unpinned_request_ids": ["r1"],
                    "remaining_pinned_requests": [],
                },
            ),
        ]
    )

    results = run_checks(
        base_url="http://localhost:8003",
        timeout=2.5,
        unpin_request_ids=["r1"],
        session=session,
    )

    assert [item["path"] for item in results] == [
        "/v1/semantic/dummy",
        "/v1/semantic/healthz",
        "/v1/semantic/pinned",
        "/v1/semantic/scheduler_state",
        "/v1/semantic/pinned/unpin",
    ]
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["json"] is None
    assert session.calls[4]["json"] == {"request_ids": ["r1"]}
    assert results[0]["body"] == {"budget": 123}
    assert results[4]["body"]["unpinned_request_ids"] == ["r1"]


def test_run_checks_sends_empty_unpin_payload_when_no_ids():
    session = FakeSession(
        [
            FakeResponse(200, {"budget": 1}),
            FakeResponse(200, {"status": "ok"}),
            FakeResponse(200, {"count": 0, "pinned_requests": []}),
            FakeResponse(504, {"error": "engine_unresponsive"}),
            FakeResponse(
                200,
                {
                    "unpinned_count": 0,
                    "unpinned_request_ids": [],
                    "remaining_pinned_requests": [],
                },
            ),
        ]
    )

    results = run_checks(
        base_url="http://localhost:9000",
        timeout=1.0,
        unpin_request_ids=[],
        session=session,
    )

    assert session.calls[4]["json"] == {}
    assert results[3]["status_code"] == 504
    assert results[3]["body"] == {"error": "engine_unresponsive"}
