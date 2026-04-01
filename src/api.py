from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.translator import translate_content


class TranslateResponse(BaseModel):
    is_english: bool
    translated_content: str


app = FastAPI()


@app.get("/")
def translator_root(content: str = Query(default="")) -> TranslateResponse:
    is_english, translated_content = translate_content(content.strip())
    return TranslateResponse(
        is_english=is_english,
        translated_content=translated_content,
    )
