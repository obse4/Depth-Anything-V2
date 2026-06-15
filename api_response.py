from typing import Any, Optional


def success_response(data: Any, message: str = "success") -> dict[str, Any]:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
    }


def error_response(code: str, message: str, data: Optional[Any] = None) -> dict[str, Any]:
    return {
        "success": False,
        "code": code,
        "message": message,
        "data": data,
    }


def validation_error_data(errors: list[dict[str, Any]]) -> dict[str, Any]:
    formatted_errors = []
    for error in errors:
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        formatted_errors.append(
            {
                "field": ".".join(location),
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            }
        )
    return {"errors": formatted_errors}
