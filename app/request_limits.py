from collections import deque

from starlette.responses import JSONResponse

from app import config


class RequestSizeLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages = deque()
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > config.MAX_REQUEST_BYTES:
                response = JSONResponse(status_code=413, content={"detail": "O envio excede o limite permitido"})
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.popleft() if messages else await receive()

        await self.app(scope, replay, send)
