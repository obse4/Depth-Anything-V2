import os
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


load_dotenv(Path(__file__).resolve().with_name(".env"))

DEFAULT_ENCODER = "vits"
DEFAULT_CHECKPOINT = "checkpoints/depth_anything_v2_vits.pth"
DEFAULT_TIMEOUT_SECONDS = 20
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class DepthRequest(BaseModel):
    image_url: str = Field(..., description="HTTP or HTTPS image URL to process.")
    oss_key: str = Field(
        default_factory=lambda: f"depth/{uuid.uuid4().hex}.png",
        description="Destination object key in OSS. Defaults to depth/<uuid>.png.",
    )

    @field_validator("image_url")
    @classmethod
    def image_url_must_be_http(cls, value: str) -> str:
        return validate_image_url(value)

    @field_validator("oss_key")
    @classmethod
    def oss_key_must_be_safe(cls, value: str) -> str:
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("oss_key must be a relative OSS object key without path traversal")
        if not value.lower().endswith(".png"):
            raise ValueError("oss_key must end with .png")
        return value


class DepthResponse(BaseModel):
    image_url: str
    oss_key: str
    depth_url: str


def validate_image_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must use http or https")
    if not parsed.netloc:
        raise ValueError("image_url must include a host")
    return url


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
    return normalized.astype(np.uint8)


def build_public_url(
    *,
    bucket_name: str,
    endpoint: str,
    object_key: str,
    public_base_url: Optional[str],
) -> str:
    encoded_key = quote(object_key, safe="/")
    if public_base_url:
        return f"{public_base_url.rstrip('/')}/{encoded_key}"

    normalized_endpoint = endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{bucket_name}.{normalized_endpoint}/{encoded_key}"


def normalize_oss_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if "://" in endpoint:
        normalized = endpoint
    else:
        normalized = f"https://{endpoint}"
    host = urlparse(normalized).netloc
    if "-internal." in host:
        raise ValueError("OSS_ENDPOINT must use the public endpoint when running outside Alibaba Cloud; remove '-internal'")
    return normalized


@dataclass
class ModelRunner:
    encoder: str = DEFAULT_ENCODER
    checkpoint_path: str = DEFAULT_CHECKPOINT
    device: Optional[str] = None
    _model: Optional[object] = field(default=None, init=False, repr=False)

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.model.infer_image(image_rgb[:, :, ::-1])

    @property
    def model(self) -> object:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> object:
        import torch

        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        if self.encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {self.encoder}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = DepthAnythingV2(**model_configs[self.encoder])
        device = self.device or _default_device(torch)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def _default_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class OssUploader:
    endpoint: str
    bucket_name: str
    access_key_id: str
    access_key_secret: str
    public_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "OssUploader":
        required = {
            "OSS_ENDPOINT": _env("OSS_ENDPOINT", "ALIYUN_OSS_ENDPOINT"),
            "OSS_BUCKET": _env("OSS_BUCKET", "ALIYUN_OSS_BUCKET"),
            "OSS_ACCESS_KEY_ID": _env("OSS_ACCESS_KEY_ID", "ALIYUN_OSS_ACCESS_KEY_ID"),
            "OSS_ACCESS_KEY_SECRET": _env("OSS_ACCESS_KEY_SECRET", "ALIYUN_OSS_ACCESS_KEY_SECRET"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing OSS environment variables: {', '.join(missing)}")

        return cls(
            endpoint=normalize_oss_endpoint(required["OSS_ENDPOINT"]),
            bucket_name=required["OSS_BUCKET"],
            access_key_id=required["OSS_ACCESS_KEY_ID"],
            access_key_secret=required["OSS_ACCESS_KEY_SECRET"],
            public_base_url=_env("OSS_PUBLIC_BASE_URL", "ALIYUN_OSS_PUBLIC_BASE_URL"),
        )

    def upload_png(self, object_key: str, png_bytes: bytes) -> str:
        import oss2
        from oss2.exceptions import OssError

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            bucket.put_object(object_key, png_bytes, headers={"Content-Type": "image/png"})
        except OssError as exc:
            raise RuntimeError(_format_oss_error(exc)) from exc
        return build_public_url(
            bucket_name=self.bucket_name,
            endpoint=self.endpoint,
            object_key=object_key,
            public_base_url=self.public_base_url,
        )


@dataclass
class DepthProcessor:
    model_runner: ModelRunner
    oss_uploader: OssUploader
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "DepthProcessor":
        return cls(
            model_runner=ModelRunner(
                encoder=os.getenv("DEPTH_ENCODER", DEFAULT_ENCODER),
                checkpoint_path=os.getenv("DEPTH_CHECKPOINT", DEFAULT_CHECKPOINT),
            ),
            oss_uploader=OssUploader.from_env(),
        )

    def process_url_to_oss(self, image_url: str, oss_key: str) -> dict[str, str]:
        image_rgb = self._download_image(image_url)
        depth = self.model_runner.infer_depth(image_rgb)
        depth_png = encode_depth_png(depth)
        depth_url = self.oss_uploader.upload_png(oss_key, depth_png)
        return {"image_url": image_url, "oss_key": oss_key, "depth_url": depth_url}

    def _download_image(self, image_url: str) -> np.ndarray:
        validate_image_url(image_url)
        response = requests.get(image_url, timeout=self.timeout_seconds, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.lower().startswith("image/"):
            raise ValueError(f"URL did not return an image content type: {content_type}")

        content = response.content
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("Image is larger than the 20MB limit")

        image = Image.open(BytesIO(content)).convert("RGB")
        return np.array(image)


def encode_depth_png(depth: np.ndarray) -> bytes:
    depth_image = normalize_depth_to_uint8(depth)
    success, encoded = cv2.imencode(".png", depth_image)
    if not success:
        raise RuntimeError("Failed to encode depth map as PNG")
    return encoded.tobytes()


def _env(primary_name: str, fallback_name: str) -> Optional[str]:
    return os.getenv(primary_name) or os.getenv(fallback_name)


def _format_oss_error(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    request_id = getattr(exc, "request_id", None)
    detail_parts = []
    if status:
        detail_parts.append(f"status={status}")
    if request_id:
        detail_parts.append(f"request_id={request_id}")
    if not detail_parts:
        return "OSS upload failed"
    return f"OSS upload failed: {', '.join(detail_parts)}"
