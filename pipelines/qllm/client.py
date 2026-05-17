import time
import json
import requests
from cli_utils import parse_query_args

class SemanticQueryBuilder:
    def __init__(self, data_path: str, model_name: str | None = None):
        self.data_path = data_path
        self.model_name = model_name
        self.plan = []

    def sem_filter(self, prompt: str):
        self.plan.append({
            "op": "sem_filter",
            "args": {
                "prompt": prompt
            }
        })
        return self

    def sem_map(self, prompt: str):
        self.plan.append({
            "op": "sem_map",
            "args": {
                "prompt": prompt
            }
        })
        return self

    def sem_classify(self, classes: list[str]):
        self.plan.append({
            "op": "sem_classify",
            "args": {
                "classes": classes
            }
        })
        return self

    def sem_agg(self, instruction: str):
        self.plan.append({
            "op": "sem_agg",
            "args": {
                "instruction": instruction
            }
        })
        return self

    def sem_topk(self, instruction: str, k: int):
        self.plan.append({
            "op": "sem_topk",
            "args": {
                "instruction": instruction,
                "k": k
            }
        })
        return self

    def sem_join(self, instruction, right_table):
        self.plan.append({
            "op": "join",
            "args": {
                "instruction": instruction,
                "right_table": right_table
            }
        })
        return self

    def build(self) -> dict:
        payload = {
            "data_path": self.data_path,
            "ops": self.plan
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        return payload

    def execute(self, endpoint: str):
        payload = self.build()
        print(payload)

        start = time.perf_counter()

        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        end = time.perf_counter()
        elapsed = end - start

        response.raise_for_status()

        print(f"Request latency: {elapsed:.3f} seconds")

        return response.json(), elapsed


if __name__ == "__main__":
    model_name, endpoint = parse_query_args()

    query_builder = SemanticQueryBuilder(
        "/scratch/hojaeson_umass/kagglehub/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_50",
        model_name=model_name,
    )

    query = (
        query_builder
            .sem_filter("Is this biology relevant paper?")
            # .sem_join("Are these relevant? return True", '/scratch/hojaeson_umass/kagglehub/spsayakpaul/arxiv-paper-abstracts/versions/2/category')
            # .sem_map("Summarize the research abstract and explain how it is related to the category")
            # .sem_join("Is this a part of research paper?", '/scratch/hojaeson_umass/kagglehub/spsayakpaul/arxiv-paper-abstracts/versions/2/category')
            # .sem_map("Summarize the research abstract and explain how it is related to the category")
            # .sem_filter("Is this AI relevant paper? return True")
            
            # .sem_filter("Is the research paper related to the given category?") 
            .sem_topk("More related to AI", 5)
            .sem_agg("Summarize again")
            # .sem_classify(classes=["theoretical", "experimental"])
            # .sem_map("Summarize the research abstract and explain how it is related to the category")
            # .sem_classify(classes=["theoretical", "experimental"])
            # .sem_agg("Summarize again")
    )

    # Print JSON before sending
    payload = query.build()

    # Send request
    result, latency = query.execute(endpoint)

    print("\nServer Response:")
    print(result)
    print(f"\nTotal request time: {latency:.3f} seconds")
