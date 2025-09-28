# api_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from retreive import query_api
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

class QueryRequest(BaseModel):
    query: str
    k: int = 3

app = FastAPI(
    title="RAG Retrieval API",
    description="Simple API wrapping retreive.query_api for front-end queries",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set narrower in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")

@app.post("/api/query")
async def api_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        result = query_api(req.query, k=req.k)
        # if retreive returned an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"ok": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# (venv) ...\Scripts\rag_arXiv> uvicorn api_route:app