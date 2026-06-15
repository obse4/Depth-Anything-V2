from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api_response import error_response, success_response, validation_error_data
from depth_service import DepthProcessor, DepthRequest
from oss_video_service import OssVideoRequest, SnapshotRequest, TailClipRequest


def get_depth_processor() -> DepthProcessor:
    return DepthProcessor.from_env()


def get_video_processor() -> OssVideoRequest:
    return OssVideoRequest.from_env()


def create_app(depth_processor=None, video_processor=None) -> FastAPI:
    app = FastAPI(title="Depth Anything V2 OSS API")

    def processor_dependency():
        return depth_processor if depth_processor is not None else get_depth_processor()

    def video_processor_dependency():
        return video_processor if video_processor is not None else get_video_processor()

    @app.get("/health")
    def health():
        return success_response({"status": "ok"})

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content=error_response(
                "VALIDATION_ERROR",
                "Request validation failed",
                validation_error_data(exc.errors()),
            ),
        )

    @app.post("/depth")
    def create_depth_map(
        request: DepthRequest,
        processor: DepthProcessor = Depends(processor_dependency),
    ):
        try:
            return success_response(processor.process_url_to_oss(request.image_url, request.oss_key))
        except FileNotFoundError as exc:
            return JSONResponse(status_code=500, content=error_response("INTERNAL_ERROR", str(exc)))
        except RuntimeError as exc:
            status_code = 502 if "OSS upload failed" in str(exc) else 500
            code = "UPSTREAM_ERROR" if status_code == 502 else "INTERNAL_ERROR"
            return JSONResponse(status_code=status_code, content=error_response(code, str(exc)))
        except ValueError as exc:
            return JSONResponse(status_code=400, content=error_response("BAD_REQUEST", str(exc)))

    @app.post("/video/snapshot")
    def create_video_snapshot(
        request: SnapshotRequest,
        processor: OssVideoRequest = Depends(video_processor_dependency),
    ):
        try:
            return success_response(processor.create_snapshot(request.video_url, request.timestamp_ms, request.wait))
        except TimeoutError as exc:
            return JSONResponse(status_code=504, content=error_response("TIMEOUT", str(exc)))
        except RuntimeError as exc:
            return JSONResponse(status_code=502, content=error_response("UPSTREAM_ERROR", str(exc)))
        except ValueError as exc:
            return JSONResponse(status_code=400, content=error_response("BAD_REQUEST", str(exc)))

    @app.post("/video/tail-clip")
    def create_video_tail_clip(
        request: TailClipRequest,
        processor: OssVideoRequest = Depends(video_processor_dependency),
    ):
        try:
            return success_response(processor.create_tail_clip(request.video_url, request.wait))
        except TimeoutError as exc:
            return JSONResponse(status_code=504, content=error_response("TIMEOUT", str(exc)))
        except RuntimeError as exc:
            return JSONResponse(status_code=502, content=error_response("UPSTREAM_ERROR", str(exc)))
        except ValueError as exc:
            return JSONResponse(status_code=400, content=error_response("BAD_REQUEST", str(exc)))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
