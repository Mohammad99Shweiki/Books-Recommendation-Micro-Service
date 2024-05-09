from fastapi import FastAPI, HTTPException, Body
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel

# todo add authentication for one user
app = FastAPI()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


class EmbeddingRequestBody(BaseModel):
    content: str


@app.post('/embed_text', response_model=List[float])
async def embed_data(request: EmbeddingRequestBody):
    try:
        embedded_data = model.encode(request.content)
        return embedded_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9090)
