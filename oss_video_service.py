import base64
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import unquote, urlparse

import oss2
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from depth_service import build_public_url, normalize_oss_endpoint, validate_image_url


load_dotenv(Path(__file__).resolve().with_name(".env"))

DEFAULT_SNAPSHOT_PREFIX = "video-snapshots"
DEFAULT_CLIP_PREFIX = "video-clips"
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_TIMEOUT_SECONDS = 120.0
TAIL_CLIP_DURATION_MS = 5000
MAX_VIDEO_BYTES = 200 * 1024 * 1024


class SnapshotRequest(BaseModel):
    video_url: str = Field(..., description="HTTP or HTTPS Aliyun OSS video URL.")
    timestamp_ms: int = Field(..., ge=0, description="Snapshot timestamp in milliseconds.")
    wait: bool = Field(default=True, description="Poll OSS until the output object exists.")

    @field_validator("video_url")
    @classmethod
    def video_url_must_be_http(cls, value: str) -> str:
        return validate_image_url(value)


class TailClipRequest(BaseModel):
    video_url: str = Field(..., description="HTTP or HTTPS Aliyun OSS video URL.")
    wait: bool = Field(default=True, description="Poll OSS until the output object exists.")

    @field_validator("video_url")
    @classmethod
    def video_url_must_be_http(cls, value: str) -> str:
        return validate_image_url(value)


@dataclass
class OssVideoClient:
    endpoint: str
    bucket_name: str
    access_key_id: str
    access_key_secret: str
    region: Optional[str] = None
    public_base_url: Optional[str] = None
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "OssVideoClient":
        endpoint = _required_env("OSS_ENDPOINT", "ALIYUN_OSS_ENDPOINT")
        return cls(
            endpoint=normalize_oss_endpoint(endpoint),
            bucket_name=_required_env("OSS_BUCKET", "ALIYUN_OSS_BUCKET"),
            access_key_id=_required_env("OSS_ACCESS_KEY_ID", "ALIYUN_OSS_ACCESS_KEY_ID"),
            access_key_secret=_required_env("OSS_ACCESS_KEY_SECRET", "ALIYUN_OSS_ACCESS_KEY_SECRET"),
            region=_env("OSS_REGION", "ALIYUN_OSS_REGION"),
            public_base_url=_env("OSS_PUBLIC_BASE_URL", "ALIYUN_OSS_PUBLIC_BASE_URL"),
            poll_interval_seconds=float(os.getenv("OSS_VIDEO_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)),
            poll_timeout_seconds=float(os.getenv("OSS_VIDEO_POLL_TIMEOUT_SECONDS", DEFAULT_POLL_TIMEOUT_SECONDS)),
        )

    @property
    def bucket(self):
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        return oss2.Bucket(auth, self.endpoint, self.bucket_name, region=self.region)

    def get_video_info(self, object_key: str) -> dict:
        try:
            result = self.bucket.get_object(object_key, process="video/info")
            return json.loads(result.read().decode("utf-8"))
        except oss2.exceptions.OssError as exc:
            raise RuntimeError(format_oss_video_error(exc)) from exc

    def async_process(self, object_key: str, process: str) -> dict:
        try:
            result = self.bucket.async_process_object(object_key, process)
            return {
                "event_id": getattr(result, "event_id", ""),
                "task_id": getattr(result, "task_id", ""),
                "request_id": getattr(result, "request_id", ""),
            }
        except oss2.exceptions.OssError as exc:
            raise RuntimeError(format_oss_video_error(exc)) from exc

    def create_snapshot(self, object_key: str, process: str) -> bytes:
        try:
            result = self.bucket.get_object(object_key, process=process)
            return result.read()
        except oss2.exceptions.OssError as exc:
            raise RuntimeError(format_oss_video_error(exc)) from exc

    def upload_bytes(self, object_key: str, content: bytes, content_type: str) -> None:
        try:
            self.bucket.put_object(object_key, content, headers={"Content-Type": content_type})
        except oss2.exceptions.OssError as exc:
            raise RuntimeError(format_oss_video_error(exc)) from exc

    def wait_until_exists(self, object_key: str) -> None:
        deadline = time.monotonic() + self.poll_timeout_seconds
        while True:
            try:
                self.bucket.head_object(object_key)
                return
            except oss2.exceptions.NoSuchKey:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for OSS object: {object_key}")
                time.sleep(self.poll_interval_seconds)

    def public_url(self, object_key: str) -> str:
        return build_public_url(
            bucket_name=self.bucket_name,
            endpoint=self.endpoint,
            object_key=object_key,
            public_base_url=self.public_base_url,
        )


@dataclass
class VideoDownloader:
    timeout_seconds: int = 60
    max_bytes: int = MAX_VIDEO_BYTES

    def download(self, url: str, destination: Path) -> None:
        validate_image_url(url)
        response = requests.get(url, timeout=self.timeout_seconds, stream=True)
        response.raise_for_status()

        total = 0
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.max_bytes:
                    raise ValueError("Video is larger than the 200MB limit")
                file.write(chunk)


@dataclass
class FfmpegVideoProcessor:
    ffmpeg_path: Optional[str] = None

    def __post_init__(self):
        if self.ffmpeg_path is None:
            import imageio_ffmpeg

            self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    def get_duration_ms(self, video_path: Path) -> int:
        command = [self.ffmpeg_path, "-hide_banner", "-i", str(video_path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = f"{result.stdout}\n{result.stderr}"
        return parse_ffmpeg_duration_ms(output)

    def create_snapshot(self, video_path: Path, timestamp_ms: int, output_path: Path) -> None:
        command = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            _seconds(timestamp_ms),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        _run_ffmpeg(command, "snapshot")

    def create_tail_clip(self, video_path: Path, start_ms: int, duration_ms: int, output_path: Path) -> None:
        command = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            _seconds(start_ms),
            "-i",
            str(video_path),
            "-t",
            _seconds(duration_ms),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        _run_ffmpeg(command, "tail clip")


@dataclass
class OssVideoRequest:
    bucket_name: str
    endpoint: str
    public_base_url: Optional[str]
    oss_client: object
    downloader: object = None
    video_processor: object = None
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX
    clip_prefix: str = DEFAULT_CLIP_PREFIX

    def __post_init__(self):
        if self.downloader is None:
            self.downloader = VideoDownloader()
        if self.video_processor is None:
            self.video_processor = FfmpegVideoProcessor()

    @classmethod
    def from_env(cls) -> "OssVideoRequest":
        client = OssVideoClient.from_env()
        return cls(
            bucket_name=client.bucket_name,
            endpoint=client.endpoint,
            public_base_url=client.public_base_url,
            oss_client=client,
            downloader=VideoDownloader(
                max_bytes=int(os.getenv("OSS_VIDEO_MAX_BYTES", MAX_VIDEO_BYTES)),
            ),
            video_processor=FfmpegVideoProcessor(),
            snapshot_prefix=os.getenv("OSS_VIDEO_SNAPSHOT_PREFIX", DEFAULT_SNAPSHOT_PREFIX),
            clip_prefix=os.getenv("OSS_VIDEO_CLIP_PREFIX", DEFAULT_CLIP_PREFIX),
        )

    def create_snapshot(self, video_url: str, timestamp_ms: int, wait: bool = True) -> dict:
        object_key_from_oss_url(
            video_url,
            bucket_name=self.bucket_name,
            endpoint=self.endpoint,
            public_base_url=self.public_base_url,
        )
        target_key = f"{self.snapshot_prefix.rstrip('/')}/{uuid.uuid4().hex}.jpg"
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "source-video"
            output_path = tmp_path / "snapshot.jpg"
            self.downloader.download(video_url, source_path)
            self.video_processor.create_snapshot(source_path, timestamp_ms, output_path)
            self.oss_client.upload_bytes(target_key, output_path.read_bytes(), "image/jpeg")

        return {
            "status": "success",
            "video_url": video_url,
            "snapshot_url": self.oss_client.public_url(target_key),
            "snapshot_key": target_key,
            "timestamp_ms": timestamp_ms,
        }

    def create_tail_clip(self, video_url: str, wait: bool = True) -> dict:
        source_key = object_key_from_oss_url(
            video_url,
            bucket_name=self.bucket_name,
            endpoint=self.endpoint,
            public_base_url=self.public_base_url,
        )
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "source-video"
            output_path = tmp_path / "tail.mp4"
            self.downloader.download(video_url, source_path)
            duration_ms = self.video_processor.get_duration_ms(source_path)
            if duration_ms <= TAIL_CLIP_DURATION_MS:
                return {
                    "status": "skipped",
                    "duration_ms": duration_ms,
                    "video_url": video_url,
                    "clip_url": video_url,
                    "clip_key": source_key,
                    "is_original": True,
                    "reason": "duration_not_more_than_5s",
                }

            target_key = f"{self.clip_prefix.rstrip('/')}/{uuid.uuid4().hex}.mp4"
            start_ms = max(0, duration_ms - TAIL_CLIP_DURATION_MS)
            self.video_processor.create_tail_clip(source_path, start_ms, TAIL_CLIP_DURATION_MS, output_path)
            self.oss_client.upload_bytes(target_key, output_path.read_bytes(), "video/mp4")

        return {
            "status": "success",
            "duration_ms": duration_ms,
            "video_url": video_url,
            "clip_url": self.oss_client.public_url(target_key),
            "clip_key": target_key,
            "is_original": False,
            "reason": None,
        }

    def get_duration_ms(self, object_key: str) -> int:
        return extract_duration_ms(json.dumps(self.oss_client.get_video_info(object_key)))


def object_key_from_oss_url(
    video_url: str,
    *,
    bucket_name: str,
    endpoint: str,
    public_base_url: Optional[str],
) -> str:
    validate_image_url(video_url)
    parsed = urlparse(video_url)

    if public_base_url and _url_is_under_base(video_url, public_base_url):
        base_path = urlparse(public_base_url).path.rstrip("/")
        object_path = parsed.path
        key = object_path[len(base_path) :].lstrip("/") if base_path else object_path.lstrip("/")
        return unquote(key)

    endpoint_host = urlparse(normalize_oss_endpoint(endpoint)).netloc
    expected_host = f"{bucket_name}.{endpoint_host}"
    if parsed.netloc != expected_host:
        raise ValueError("video_url must point to the current OSS bucket")

    return unquote(parsed.path.lstrip("/"))


def extract_duration_ms(video_info_json: str) -> int:
    payload = json.loads(video_info_json)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        duration = payload.get("Format", {}).get("Duration")
    if duration is None:
        raise ValueError("video/info response did not include duration")
    return int(float(duration) * 1000)


def parse_ffmpeg_duration_ms(output: str) -> int:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError("Unable to read video duration from ffmpeg output")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return int((hours * 3600 + minutes * 60 + seconds) * 1000)


def build_saveas_process(style: str, bucket_name: str, target_key: str) -> str:
    encoded_bucket = _base64url(bucket_name)
    encoded_target = _base64url(target_key)
    return f"{style}|sys/saveas,b_{encoded_bucket},o_{encoded_target}"


def _base64url(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _url_is_under_base(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(base_url)
    if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
        return False
    return parsed.path == base.path or parsed.path.startswith(base.path.rstrip("/") + "/")


def _env(primary_name: str, fallback_name: str) -> Optional[str]:
    return os.getenv(primary_name) or os.getenv(fallback_name)


def _required_env(primary_name: str, fallback_name: str) -> str:
    value = _env(primary_name, fallback_name)
    if not value:
        raise RuntimeError(f"Missing OSS environment variable: {primary_name}")
    return value


def format_oss_video_error(exc: Exception) -> str:
    detail_parts = []
    status = getattr(exc, "status", None)
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    request_id = getattr(exc, "request_id", None)
    ec = getattr(exc, "ec", None)
    if status:
        detail_parts.append(f"status={status}")
    if code:
        detail_parts.append(f"code={code}")
    if message:
        detail_parts.append(f"message={message}")
    if request_id:
        detail_parts.append(f"request_id={request_id}")
    if ec:
        detail_parts.append(f"ec={ec}")
    if not detail_parts:
        return "OSS video processing failed"
    return f"OSS video processing failed: {', '.join(detail_parts)}"


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _run_ffmpeg(command: list[str], operation: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg {operation} failed: {stderr}")
