from fastapi.testclient import TestClient

import api


class FakeDepthProcessor:
    def process_url_to_oss(self, image_url, oss_key):
        return {
            "image_url": image_url,
            "oss_key": oss_key,
            "depth_url": f"https://cdn.example.com/{oss_key}",
        }


class FailingDepthProcessor:
    def process_url_to_oss(self, image_url, oss_key):
        raise RuntimeError("OSS upload failed: status=502")


class FailingVideoProcessor:
    def create_snapshot(self, video_url, timestamp_ms, wait):
        raise RuntimeError("OSS video processing failed: status=404, code=Imm Client")

    def create_tail_clip(self, video_url, wait):
        raise RuntimeError("OSS video processing failed: status=404, code=Imm Client")


class FakeVideoProcessor:
    def create_snapshot(self, video_url, timestamp_ms, wait):
        return {
            "status": "success",
            "video_url": video_url,
            "snapshot_url": "https://cdn.example.com/video-snapshots/a.jpg",
            "timestamp_ms": timestamp_ms,
            "wait": wait,
        }

    def create_tail_clip(self, video_url, wait):
        return {
            "status": "success",
            "video_url": video_url,
            "clip_url": "https://cdn.example.com/video-clips/a.mp4",
            "duration_ms": 9000,
            "wait": wait,
        }


def test_depth_endpoint_processes_url_and_returns_oss_location():
    app = api.create_app(depth_processor=FakeDepthProcessor())
    client = TestClient(app)

    response = client.post(
        "/depth",
        json={
            "image_url": "https://example.com/input.jpg",
            "oss_key": "depth/result.png",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "code": "OK",
        "message": "success",
        "data": {
            "image_url": "https://example.com/input.jpg",
            "oss_key": "depth/result.png",
            "depth_url": "https://cdn.example.com/depth/result.png",
        },
    }


def test_depth_endpoint_rejects_non_http_url():
    app = api.create_app(depth_processor=FakeDepthProcessor())
    client = TestClient(app)

    response = client.post("/depth", json={"image_url": "file:///tmp/input.jpg"})

    assert response.status_code == 422
    assert response.json()["success"] is False
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.json()["data"]["errors"][0]["field"] == "image_url"


def test_depth_endpoint_reports_upload_failures_as_bad_gateway():
    app = api.create_app(depth_processor=FailingDepthProcessor())
    client = TestClient(app)

    response = client.post(
        "/depth",
        json={
            "image_url": "https://example.com/input.jpg",
            "oss_key": "depth/result.png",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "code": "UPSTREAM_ERROR",
        "message": "OSS upload failed: status=502",
        "data": None,
    }


def test_video_snapshot_endpoint_returns_snapshot_url():
    app = api.create_app(depth_processor=FakeDepthProcessor(), video_processor=FakeVideoProcessor())
    client = TestClient(app)

    response = client.post(
        "/video/snapshot",
        json={"video_url": "https://cdn.example.com/videos/a.mp4", "timestamp_ms": 1200},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"]["snapshot_url"] == "https://cdn.example.com/video-snapshots/a.jpg"


def test_video_tail_clip_endpoint_returns_clip_url():
    app = api.create_app(depth_processor=FakeDepthProcessor(), video_processor=FakeVideoProcessor())
    client = TestClient(app)

    response = client.post(
        "/video/tail-clip",
        json={"video_url": "https://cdn.example.com/videos/a.mp4"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"]["clip_url"] == "https://cdn.example.com/video-clips/a.mp4"


def test_video_snapshot_endpoint_reports_oss_failures_as_bad_gateway():
    app = api.create_app(depth_processor=FakeDepthProcessor(), video_processor=FailingVideoProcessor())
    client = TestClient(app)

    response = client.post(
        "/video/snapshot",
        json={"video_url": "https://cdn.example.com/videos/a.mp4", "timestamp_ms": 1200},
    )

    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "code": "UPSTREAM_ERROR",
        "message": "OSS video processing failed: status=404, code=Imm Client",
        "data": None,
    }
