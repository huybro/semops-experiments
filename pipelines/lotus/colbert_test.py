import argparse
import csv
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")

DEFAULT_CORPUS_CSV = os.path.join(PROJECT_ROOT, "data", "beir_fever_corpus_data.csv")
DEFAULT_INDEX_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "colbert_indexes",
)
DEFAULT_INDEX_NAME = "fever_factool_wikipedia_colbert"
DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"
DEFAULT_COLBERT_ROOT = os.path.join(PROJECT_ROOT, "projects", "ColBERT")


def raise_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reusable ColBERT index.")
    parser.add_argument("--corpus-csv", default=os.environ.get("FEVER_CORPUS_CSV", DEFAULT_CORPUS_CSV))
    parser.add_argument("--text-column", default=os.environ.get("COLBERT_TEXT_COLUMN", "data"))
    parser.add_argument("--id-column", default=os.environ.get("COLBERT_ID_COLUMN"))
    parser.add_argument("--index-root", default=os.environ.get("COLBERT_INDEX_ROOT", DEFAULT_INDEX_ROOT))
    parser.add_argument("--index-name", default=os.environ.get("COLBERT_INDEX_NAME", DEFAULT_INDEX_NAME))
    parser.add_argument("--checkpoint", default=os.environ.get("COLBERT_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--colbert-root", default=os.environ.get("COLBERT_ROOT", DEFAULT_COLBERT_ROOT))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("FEVER_CORPUS_LIMIT", "0")))
    parser.add_argument("--nbits", type=int, default=int(os.environ.get("COLBERT_NBITS", "2")))
    parser.add_argument("--nranks", type=int, default=int(os.environ.get("COLBERT_NRANKS", "1")))
    parser.add_argument("--experiment", default=os.environ.get("COLBERT_EXPERIMENT", "wikipedia"))
    parser.add_argument("--collection", default=os.environ.get("COLBERT_COLLECTION"))
    parser.add_argument(
        "--overwrite",
        default=os.environ.get("COLBERT_OVERWRITE_INDEX", "reuse"),
        choices=("reuse", "resume", "force_silent_overwrite"),
    )
    return parser.parse_args()


def write_collection_tsv(corpus_csv, collection_path, text_column, id_column, limit):
    os.makedirs(os.path.dirname(collection_path), exist_ok=True)

    count = 0
    with open(corpus_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if text_column not in reader.fieldnames:
            raise ValueError(f"{corpus_csv} has no column named {text_column!r}")
        if id_column and id_column not in reader.fieldnames:
            raise ValueError(f"{corpus_csv} has no column named {id_column!r}")

        with open(collection_path, "w", encoding="utf-8") as tsv_file:
            for row in reader:
                if limit > 0 and count >= limit:
                    break
                doc_id = row[id_column] if id_column else str(count)
                text = " ".join((row.get(text_column) or "").split())
                tsv_file.write(f"{doc_id}\t{text}\n")
                count += 1

    return count


def add_colbert_repo_to_path(colbert_root):
    if not colbert_root:
        return

    colbert_root = os.path.abspath(colbert_root)
    if not os.path.isdir(colbert_root):
        raise FileNotFoundError(
            f"ColBERT repo not found at {colbert_root}. Clone it or pass --colbert-root /path/to/ColBERT."
        )
    sys.path.insert(0, colbert_root)


def build_index(args):
    add_colbert_repo_to_path(args.colbert_root)

    import faiss
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    _ = faiss

    os.makedirs(args.index_root, exist_ok=True)
    collection = args.collection or os.path.join(args.index_root, "collections", "wikipedia.tsv")
    if not args.collection:
        count = write_collection_tsv(
            corpus_csv=args.corpus_csv,
            collection_path=collection,
            text_column=args.text_column,
            id_column=args.id_column,
            limit=args.limit,
        )
        print(f"Wrote {count} documents to {collection}")

    with Run().context(
        RunConfig(nranks=args.nranks, experiment=args.experiment, root=args.index_root)
    ):
        config = ColBERTConfig(
            nbits=args.nbits,
            root=args.index_root,
        )
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name, collection=collection, overwrite=args.overwrite)


def main():
    raise_csv_field_limit()
    build_index(parse_args())


if __name__ == "__main__":
    main()
