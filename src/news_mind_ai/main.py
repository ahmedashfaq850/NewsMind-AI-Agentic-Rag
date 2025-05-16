from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List

from news_mind_ai.agents_team import run_agent_pipeline
from news_mind_ai.models import ArticleOutput

app = FastAPI(
    title="News Mind AI",
    description="API for processing queries and generating news articles using an agent pipeline.",
    version="0.1.0",
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class QueryModel(BaseModel):
    query: str


class SuccessResponse(BaseModel):
    status_code: int = status.HTTP_201_CREATED
    message: str
    data: ArticleOutput


class ErrorResponse(BaseModel):
    status_code: int
    message: str
    detail: Any = None


@app.post(
    "/generate-article",
    response_model=SuccessResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Bad request e.g. if the query is empty",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error",
        },
    },
)
async def generate_article(request: QueryModel):
    if not request.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    try:
        pipeline_result = await run_agent_pipeline(request.query)
        return SuccessResponse(
            message="Query processed successfully and article generated.",
            data=pipeline_result,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
