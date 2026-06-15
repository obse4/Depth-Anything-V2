from api_response import error_response, success_response


def test_success_response_uses_standard_envelope():
    response = success_response({"url": "https://example.com/a.png"})

    assert response == {
        "success": True,
        "code": "OK",
        "message": "success",
        "data": {"url": "https://example.com/a.png"},
    }


def test_error_response_uses_standard_envelope():
    response = error_response("BAD_REQUEST", "invalid input", {"field": "url"})

    assert response == {
        "success": False,
        "code": "BAD_REQUEST",
        "message": "invalid input",
        "data": {"field": "url"},
    }
