import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ColbertWiki import ColbertWiki


class BuildIndexRequest(BaseModel):
    cp_id: str
    right_table: list[list]


class QueryRequest(BaseModel):
    cp_id: str
    left_tuple: list
    right_table: list[list] | None = None
    top_k: int | None = Field(default=None, gt=0)
    low_threshold: float | None = None
    high_threshold: float | None = None


class ClearRequest(BaseModel):
    cp_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.wiki = ColbertWiki(
        index_name=app.state.index_name,
        experiment_root=app.state.experiment_root,
        experiment=app.state.experiment,
        collection=app.state.collection,
        colbert_root=app.state.colbert_root,
    )
    yield


def create_app(
    index_name: str,
    experiment_root: str,
    experiment: str,
    collection: str,
    colbert_root: str,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.index_name = index_name
    app.state.experiment_root = experiment_root
    app.state.experiment = experiment
    app.state.collection = collection
    app.state.colbert_root = colbert_root

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "colbert"}

    @app.get("/cache")
    async def cache():
        return {"cache": {app.state.index_name: len(app.state.wiki.searcher.collection)}}

    @app.post("/build_index")
    async def build_index(request: BuildIndexRequest):
        return {
            "rows_indexed": len(app.state.wiki.searcher.collection),
            "cp_id": request.cp_id,
            "backend": "colbert",
            "note": "Using prebuilt ColBERT index; right_table ignored.",
        }

    @app.post("/query")
    async def query(request: QueryRequest):
        top_k = request.top_k or 10
        query_text = " ".join(map(str, request.left_tuple))
        results = [
            {
                "metadata": [result["pid"], result["text"]],
                "score": result["score"],
            }
            for result in app.state.wiki.search(query_text, topk=top_k)
        ]
        return {"results": results}

    @app.post("/clear")
    async def clear(request: ClearRequest):
        return {"cleared": False, "cp_id": request.cp_id, "backend": "colbert"}

    return app


def main():
    parser = argparse.ArgumentParser(description="ICP-compatible service backed by ColBERT.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--index-name", default="fever_factool_wikipedia_colbert")
    parser.add_argument(
        "--experiment-root",
        default="/home/hojaeson_umass/projects/semops-experiments/pipelines/lotus/logs/colbert_indexes",
    )
    parser.add_argument("--experiment", default="wikipedia")
    parser.add_argument(
        "--collection",
        default="/home/hojaeson_umass/projects/semops-experiments/pipelines/lotus/logs/colbert_indexes/collections/wikipedia.tsv",
    )
    parser.add_argument(
        "--colbert-root",
        default="/home/hojaeson_umass/projects/semops-experiments/projects/ColBERT",
    )
    args = parser.parse_args()

    app = create_app(
        index_name=args.index_name,
        experiment_root=args.experiment_root,
        experiment=args.experiment,
        collection=args.collection,
        colbert_root=args.colbert_root,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
