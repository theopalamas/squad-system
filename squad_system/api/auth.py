import os

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer


class ApiKeyBearer(HTTPBearer):
    def __init__(self, secret: str, auto_error: bool = True):
        super(ApiKeyBearer, self).__init__(auto_error=auto_error)
        self._secret = secret

    async def __call__(self, request: Request) -> str:
        credentials = await super(ApiKeyBearer, self).__call__(request)

        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

        if not credentials.scheme == "Bearer":
            raise HTTPException(
                status_code=403, detail="Invalid authentication scheme."
            )

        if not credentials.credentials == self._secret:
            raise HTTPException(status_code=403, detail="Invalid API key")

        return credentials.credentials


def create_api_key_bearer() -> ApiKeyBearer:
    return ApiKeyBearer(secret=os.environ["API_KEY"])
