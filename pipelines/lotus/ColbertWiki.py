import argparse
import csv
import os
import sys
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\..*is deprecated.*",
    category=FutureWarning,
)


PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
DEFAULT_COLBERT_ROOT = os.path.join(PROJECT_ROOT, "projects", "ColBERT")
DEFAULT_EXPERIMENT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "colbert_indexes",
)
DEFAULT_EXPERIMENT = "wikipedia"
DEFAULT_INDEX_NAME = "fever_factool_wikipedia_colbert"
DEFAULT_COLLECTION = os.path.join(DEFAULT_EXPERIMENT_ROOT, "collections", "wikipedia.tsv")
DEFAULT_ICP_ADDRESS = os.environ.get("ICP_ADDRESS", "127.0.0.1")
DEFAULT_ICP_PORT = int(os.environ.get("ICP_PORT", "8080"))
DEFAULT_ICP_CP_ID = os.environ.get("ICP_CP_ID", "fever_wikipedia")
DEFAULT_ICP_LIMIT = int(os.environ.get("ICP_LIMIT", "10000"))
DEFAULT_ICP_BUILD = os.environ.get("ICP_BUILD", "1") == "1"


def add_colbert_repo_to_path(colbert_root=DEFAULT_COLBERT_ROOT):
    colbert_root = os.path.abspath(colbert_root)
    if not os.path.isdir(colbert_root):
        raise FileNotFoundError(f"ColBERT repo not found at {colbert_root}")
    if colbert_root not in sys.path:
        sys.path.insert(0, colbert_root)


class ColbertWiki:
    def __init__(
        self,
        index_name=DEFAULT_INDEX_NAME,
        experiment_root=DEFAULT_EXPERIMENT_ROOT,
        experiment=DEFAULT_EXPERIMENT,
        collection=DEFAULT_COLLECTION,
        colbert_root=DEFAULT_COLBERT_ROOT,
    ):
        add_colbert_repo_to_path(colbert_root)

        from colbert import Searcher
        from colbert.infra import Run, RunConfig

        self.index_name = index_name
        self.experiment_root = experiment_root
        self.experiment = experiment
        self.collection = collection

        with Run().context(RunConfig(root=experiment_root, experiment=experiment)):
            self.searcher = Searcher(index=index_name, collection=collection)

    def search(self, query, topk=10):
        pids, ranks, scores = self.searcher.search(query, k=topk)
        results = []
        for pid, rank, score in zip(pids, ranks, scores):
            pid = int(pid)
            results.append(
                {
                    "pid": pid,
                    "rank": int(rank),
                    "score": float(score),
                    "text": self.searcher.collection[pid],
                }
            )
        return results

    def to_lotus_dataframe(self):
        df = pd.DataFrame({"content": self.searcher.collection.data})
        df.attrs["index_dirs"] = {"content": self.index_name}
        return df

    def lotus_settings(self):
        return ColbertWikiRM(), ColbertWikiVS(self)


def load_tsv_rows(collection, limit):
    rows = []
    with open(collection, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit is not None and limit > 0 and i >= limit:
                break
            if len(row) < 2:
                continue
            rows.append((int(row[0]), row[1]))
    return rows


class IcpWiki:
    def __init__(
        self,
        collection=DEFAULT_COLLECTION,
        cp_id=DEFAULT_ICP_CP_ID,
        service_address=DEFAULT_ICP_ADDRESS,
        service_port=DEFAULT_ICP_PORT,
        limit=DEFAULT_ICP_LIMIT,
        build=DEFAULT_ICP_BUILD,
    ):
        self.collection = collection
        self.cp_id = cp_id
        self.index_name = cp_id
        self.base_url = f"http://{service_address}:{service_port}"
        self.rows = load_tsv_rows(collection, limit)
        self.pid_to_text = {pid: text for pid, text in self.rows}

        if build:
            self.build_index()

    def build_index(self):
        response = requests.post(
            f"{self.base_url}/build_index",
            json={
                "cp_id": self.cp_id,
                "right_table": [[pid, text] for pid, text in self.rows],
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def search(self, query, topk=10):
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "cp_id": self.cp_id,
                "left_tuple": [query],
                "top_k": topk,
            },
            timeout=120,
        )
        response.raise_for_status()

        results = []
        for rank, item in enumerate(response.json()["results"], start=1):
            metadata = item["metadata"]
            pid = int(metadata[0])
            text = self.pid_to_text.get(pid, metadata[1] if len(metadata) > 1 else "")
            results.append(
                {
                    "pid": pid,
                    "rank": rank,
                    "score": float(item["score"]),
                    "text": text,
                }
            )
        return results

    def to_lotus_dataframe(self):
        df = pd.DataFrame(
            {"content": [text for _, text in self.rows]},
            index=[pid for pid, _ in self.rows],
        )
        df.attrs["index_dirs"] = {"content": self.index_name}
        return df

    def lotus_settings(self):
        return ColbertWikiRM(), ColbertWikiVS(self)


class ColbertWikiRM:
    def _embed(self, docs):
        return np.asarray(docs, dtype=object)

    def __call__(self, docs):
        return self._embed(docs)

    def convert_query_to_query_vector(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        return np.asarray(queries, dtype=object)


class ColbertWikiVS:
    def __init__(self, wiki):
        self.wiki = wiki
        self.index_dir = wiki.index_name

    def index(self, docs, embeddings, index_dir, **kwargs):
        self.index_dir = index_dir

    def load_index(self, index_dir):
        self.index_dir = index_dir

    def get_vectors_from_index(self, index_dir, ids):
        raise NotImplementedError("ColBERT does not expose Lotus-style vectors.")

    def __call__(self, query_vectors, K, ids=None, **kwargs):
        from lotus.types import RMOutput

        distances = []
        indices = []
        for query in query_vectors.tolist():
            results = self.wiki.search(str(query), topk=K)
            if ids is not None:
                allowed = set(ids)
                results = [result for result in results if result["pid"] in allowed]
            indices.append([result["pid"] for result in results])
            distances.append([result["score"] for result in results])
        return RMOutput(distances=distances, indices=indices)


def parse_args():
    parser = argparse.ArgumentParser(description="Search the indexed FEVER Wikipedia corpus.")
    parser.add_argument("query")
    parser.add_argument("--backend", choices=("colbert", "icp"), default="colbert")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--colbert-root", default=DEFAULT_COLBERT_ROOT)
    parser.add_argument("--icp-address", default=DEFAULT_ICP_ADDRESS)
    parser.add_argument("--icp-port", type=int, default=DEFAULT_ICP_PORT)
    parser.add_argument("--icp-cp-id", default=DEFAULT_ICP_CP_ID)
    parser.add_argument("--icp-limit", type=int, default=DEFAULT_ICP_LIMIT)
    parser.add_argument("--icp-build", action="store_true", default=DEFAULT_ICP_BUILD)
    parser.add_argument("--icp-no-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "icp":
        wiki = IcpWiki(
            collection=args.collection,
            cp_id=args.icp_cp_id,
            service_address=args.icp_address,
            service_port=args.icp_port,
            limit=args.icp_limit,
            build=args.icp_build and not args.icp_no_build,
        )
    else:
        wiki = ColbertWiki(
            index_name=args.index_name,
            experiment_root=args.experiment_root,
            experiment=args.experiment,
            collection=args.collection,
            colbert_root=args.colbert_root,
        )
    for result in wiki.search(args.query, topk=args.topk):
        print(f"{result['rank']}\t{result['pid']}\t{result['score']:.4f}\t{result['text']}")


if __name__ == "__main__":
    main()
