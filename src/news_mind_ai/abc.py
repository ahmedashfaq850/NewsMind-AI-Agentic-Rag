from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Any, Dict, List

from src.news_mind_ai.agents_team import run_agent_pipeline
from src.news_mind_ai.models import ArticleOutput  # For response model typing

app = FastAPI(
    title="News Mind AI",
    description="API for processing queries and generating news articles using an agent pipeline.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
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
    status_code=status.HTTP_201_CREATED,
    summary="Process a user query to generate a news article",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Bad request e.g. if pipeline fails to produce output",
        },
    },
)
async def process_query_endpoint(request: QueryRequest):

    try:
        if not request.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty",
            )
        pipeline_result = await run_agent_pipeline(request.query)
        return SuccessResponse(
            message="Query processed successfully and article generated.",
            data=pipeline_result,
        )

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        # Fallback for any other unhandled exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
