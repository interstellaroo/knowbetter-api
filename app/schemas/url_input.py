from pydantic import BaseModel, HttpUrl


class UrlInputSchema(BaseModel):
    url: HttpUrl