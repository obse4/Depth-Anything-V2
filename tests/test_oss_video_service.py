import json
from pathlib import Path

import pytest

from oss_video_service import (
    FfmpegVideoProcessor,
    OssVideoRequest,
    parse_ffmpeg_duration_ms,
    extract_duration_ms,
    object_key_from_oss_url,
)


class FakeOssClient:
    def __init__(self, duration_ms=8000):
        self.duration_ms = duration_ms
        self.async_calls = []
        self.head_calls = []
        self.upload_calls = []

    def get_video_info(self, object_key):
        return {"format": {"duration": self.duration_ms / 1000}}

    def async_process(self, object_key, process):
        self.async_calls.append((object_key, process))
        return {"event_id": "event-1", "task_id": "task-1", "request_id": "request-1"}

    def upload_bytes(self, object_key, content, content_type):
        self.upload_calls.append((object_key, content, content_type))

    def wait_until_exists(self, object_key):
        self.head_calls.append(object_key)

    def public_url(self, object_key):
        return f"https://cdn.example.com/{object_key}"


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def download(self, url, destination):
        self.calls.append((url, destination))
        destination.write_bytes(b"video")


class FakeVideoProcessor:
    def __init__(self, duration_ms=8000):
        self.duration_ms = duration_ms
        self.snapshot_calls = []
        self.tail_calls = []

    def get_duration_ms(self, video_path):
        return self.duration_ms

    def create_snapshot(self, video_path, timestamp_ms, output_path):
        self.snapshot_calls.append((video_path, timestamp_ms, output_path))
        output_path.write_bytes(b"jpg-bytes")

    def create_tail_clip(self, video_path, start_ms, duration_ms, output_path):
        self.tail_calls.append((video_path, start_ms, duration_ms, output_path))
        output_path.write_bytes(b"mp4-bytes")


def test_object_key_from_current_bucket_url():
    key = object_key_from_oss_url(
        "https://demo-bucket.oss-cn-beijing.aliyuncs.com/videos/a.mp4",
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        public_base_url=None,
    )

    assert key == "videos/a.mp4"


def test_object_key_from_public_base_url():
    key = object_key_from_oss_url(
        "https://cdn.example.com/media/videos/a.mp4",
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        public_base_url="https://cdn.example.com/media",
    )

    assert key == "videos/a.mp4"


def test_object_key_rejects_other_bucket_url():
    with pytest.raises(ValueError, match="current OSS bucket"):
        object_key_from_oss_url(
            "https://other-bucket.oss-cn-beijing.aliyuncs.com/videos/a.mp4",
            bucket_name="demo-bucket",
            endpoint="https://oss-cn-beijing.aliyuncs.com",
            public_base_url=None,
        )


def test_extract_duration_ms_from_video_info():
    payload = {"format": {"duration": "5.432"}}

    assert extract_duration_ms(json.dumps(payload)) == 5432


def test_parse_ffmpeg_duration_ms():
    stderr = "Duration: 00:01:02.34, start: 0.000000, bitrate: 123 kb/s"

    assert parse_ffmpeg_duration_ms(stderr) == 62340


def test_snapshot_downloads_video_uses_ffmpeg_then_uploads_image():
    downloader = FakeDownloader()
    processor = FakeVideoProcessor()
    request = OssVideoRequest(
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        public_base_url="https://cdn.example.com",
        oss_client=FakeOssClient(),
        downloader=downloader,
        video_processor=processor,
    )

    response = request.create_snapshot(
        video_url="https://cdn.example.com/videos/a.mp4",
        timestamp_ms=1200,
        wait=True,
    )

    assert response["status"] == "success"
    assert response["snapshot_url"].startswith("https://cdn.example.com/video-snapshots/")
    assert downloader.calls[0][0] == "https://cdn.example.com/videos/a.mp4"
    assert processor.snapshot_calls[0][1] == 1200
    assert request.oss_client.upload_calls[0][0].startswith("video-snapshots/")
    assert request.oss_client.upload_calls[0][1] == b"jpg-bytes"
    assert request.oss_client.upload_calls[0][2] == "image/jpeg"
    assert request.oss_client.async_calls == []


def test_tail_clip_returns_original_url_when_duration_is_not_more_than_5_seconds():
    downloader = FakeDownloader()
    request = OssVideoRequest(
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        public_base_url="https://cdn.example.com",
        oss_client=FakeOssClient(duration_ms=5000),
        downloader=downloader,
        video_processor=FakeVideoProcessor(duration_ms=5000),
    )

    response = request.create_tail_clip("https://cdn.example.com/videos/a.mp4", wait=True)

    assert response == {
        "status": "skipped",
        "duration_ms": 5000,
        "video_url": "https://cdn.example.com/videos/a.mp4",
        "clip_url": "https://cdn.example.com/videos/a.mp4",
        "clip_key": "videos/a.mp4",
        "is_original": True,
        "reason": "duration_not_more_than_5s",
    }
    assert request.oss_client.async_calls == []
    assert downloader.calls[0][0] == "https://cdn.example.com/videos/a.mp4"


def test_tail_clip_downloads_video_uses_ffmpeg_then_uploads_clip():
    downloader = FakeDownloader()
    processor = FakeVideoProcessor(duration_ms=8750)
    request = OssVideoRequest(
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        public_base_url="https://cdn.example.com",
        oss_client=FakeOssClient(duration_ms=8750),
        downloader=downloader,
        video_processor=processor,
    )

    response = request.create_tail_clip("https://cdn.example.com/videos/a.mp4", wait=True)

    assert response["status"] == "success"
    assert response["duration_ms"] == 8750
    assert response["is_original"] is False
    assert response["reason"] is None
    assert response["clip_url"].startswith("https://cdn.example.com/video-clips/")
    assert processor.tail_calls[0][1] == 3750
    assert processor.tail_calls[0][2] == 5000
    assert request.oss_client.upload_calls[0][0].startswith("video-clips/")
    assert request.oss_client.upload_calls[0][1] == b"mp4-bytes"
    assert request.oss_client.upload_calls[0][2] == "video/mp4"
    assert request.oss_client.async_calls == []


def test_ffmpeg_processor_uses_project_ffmpeg_binary():
    processor = FfmpegVideoProcessor()

    assert Path(processor.ffmpeg_path).name.startswith("ffmpeg")
