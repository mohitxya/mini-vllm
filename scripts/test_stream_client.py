import json
import requests


def main():
    url = "http://127.0.0.1:8000/generate_stream"

    payload = {
        "prompt": "A GPU is useful because",
        "max_new_tokens": 40,
        "strategy": "top_k",
        "temperature": 0.8,
        "top_k": 50,
        "seed": 42,
    }

    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()

        print("\n--- Streaming output ---\n")

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            event = json.loads(line)

            if event["type"] == "token":
                print(event["text"], end="", flush=True)

            elif event["type"] == "done":
                print("\n\n--- Done ---")
                print("Generated tokens:", event["num_generated_tokens"])
                print("Total time:", event["total_time_seconds"])

            elif event["type"] == "error":
                print("\nError:", event["message"])


if __name__ == "__main__":
    main()