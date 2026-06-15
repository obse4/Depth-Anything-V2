import numpy as np
import pytest

from depth_service import (
    DepthRequest,
    OssUploader,
    build_public_url,
    normalize_depth_to_uint8,
    normalize_oss_endpoint,
    validate_image_url,
)


def test_validate_image_url_accepts_http_and_https():
    assert validate_image_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert validate_image_url("http://example.com/a.jpg") == "http://example.com/a.jpg"


@pytest.mark.parametrize("url", ["ftp://example.com/a.jpg", "file:///tmp/a.jpg", "example.com/a.jpg"])
def test_validate_image_url_rejects_non_http_urls(url):
    with pytest.raises(ValueError, match="http"):
        validate_image_url(url)


def test_depth_request_generates_png_key_when_not_provided():
    request = DepthRequest(image_url="https://example.com/a.jpg")

    assert request.oss_key.startswith("depth/")
    assert request.oss_key.endswith(".png")


def test_depth_request_rejects_path_traversal_key():
    with pytest.raises(ValueError, match="oss_key"):
        DepthRequest(image_url="https://example.com/a.jpg", oss_key="../secret.png")


def test_normalize_depth_to_uint8_scales_dynamic_range():
    depth = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)

    normalized = normalize_depth_to_uint8(depth)

    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_normalize_depth_to_uint8_handles_flat_depth_map():
    depth = np.full((2, 2), 7.0, dtype=np.float32)

    normalized = normalize_depth_to_uint8(depth)

    assert normalized.tolist() == [[0, 0], [0, 0]]


def test_build_public_url_uses_explicit_base_url():
    url = build_public_url(
        bucket_name="demo-bucket",
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        object_key="depth/a.png",
        public_base_url="https://cdn.example.com/prefix/",
    )

    assert url == "https://cdn.example.com/prefix/depth/a.png"


def test_build_public_url_uses_bucket_endpoint_when_no_base_url():
    url = build_public_url(
        bucket_name="demo-bucket",
        endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        object_key="depth/a.png",
        public_base_url=None,
    )

    assert url == "https://demo-bucket.oss-cn-hangzhou.aliyuncs.com/depth/a.png"


def test_normalize_oss_endpoint_adds_https_when_scheme_is_missing():
    assert normalize_oss_endpoint("oss-cn-hangzhou.aliyuncs.com") == "https://oss-cn-hangzhou.aliyuncs.com"


def test_normalize_oss_endpoint_preserves_existing_scheme():
    assert normalize_oss_endpoint("https://oss-cn-hangzhou.aliyuncs.com") == "https://oss-cn-hangzhou.aliyuncs.com"


def test_normalize_oss_endpoint_rejects_internal_endpoint():
    with pytest.raises(ValueError, match="internal"):
        normalize_oss_endpoint("https://oss-cn-beijing-internal.aliyuncs.com")


def test_oss_uploader_reads_existing_oss_env_names(monkeypatch):
    monkeypatch.setenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
    monkeypatch.setenv("OSS_BUCKET", "demo-bucket")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "access-key-id")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "access-key-secret")
    monkeypatch.setenv("OSS_PUBLIC_BASE_URL", "https://cdn.example.com")

    uploader = OssUploader.from_env()

    assert uploader.endpoint == "https://oss-cn-hangzhou.aliyuncs.com"
    assert uploader.bucket_name == "demo-bucket"
    assert uploader.access_key_id == "access-key-id"
    assert uploader.access_key_secret == "access-key-secret"
    assert uploader.public_base_url == "https://cdn.example.com"
