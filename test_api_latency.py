import time
import csv
import statistics
import requests
import matplotlib.pyplot as plt

BASE_URL = "http://ece444pra5-env.eba-ax7jes92.us-east-2.elasticbeanstalk.com"
PREDICT_URL = f"{BASE_URL}/predict"

# 4 test inputs (2 fake, 2 real).
TEST_CASES = [
    ("fake_1", "Breaking: Celebrity endorses miracle cure that doctors hate. Click here now!"),
    ("fake_2", "Scientists shock the world: coffee cured cancer overnight, but media hides it."),
    ("real_1", "The Federal Reserve announced an interest rate change during Wednesday's meeting."),
    ("real_2", "The University of Toronto announced its fall semester will begin on September 5, with most classes held in person."),
]


def call_api(text: str) -> str:
    """Call the /predict endpoint and return the predicted label."""
    resp = requests.post(PREDICT_URL, json={"message": text}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("label", "")


def functional_tests():
    print("=== Functional / unit tests ===")
    for name, text in TEST_CASES:
        try:
            label = call_api(text)
            print(f"{name}: prediction = {label}")
        except Exception as e:
            print(f"{name}: ERROR calling API: {e}")
    print()


def latency_tests():
    print("=== Latency / performance tests ===")
    all_latencies = {}  # name -> list of 100 latencies (ms)

    for name, text in TEST_CASES:
        latencies_ms = []
        csv_filename = f"latency_{name}.csv"

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "start_time", "end_time", "latency_ms"])

            for i in range(100):
                start = time.perf_counter()
                resp = requests.post(PREDICT_URL, json={"message": text}, timeout=10)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000.0
                latencies_ms.append(latency_ms)

                # one row per call; exactly 100 rows per test case
                writer.writerow([i, start, end, latency_ms])

                if resp.status_code != 200:
                    print(f"[{name}] iter {i}: non-200 status {resp.status_code}, body={resp.text}")

        mean_latency = statistics.mean(latencies_ms)
        print(f"{name}: average latency = {mean_latency:.2f} ms over 100 calls")

        all_latencies[name] = latencies_ms

    print()
    return all_latencies


def plot_boxplot(all_latencies):
    labels = list(all_latencies.keys())
    data = [all_latencies[name] for name in labels]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Latency (ms)")
    plt.title("API Latency by Test Case (100 calls each)")
    plt.tight_layout()
    plt.savefig("latency_boxplot.png")
    print("Saved boxplot to latency_boxplot.png")


if __name__ == "__main__":
    functional_tests()
    latencies = latency_tests()
    plot_boxplot(latencies)
