from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_scalars(log_dir: Path):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags()["scalars"]:
        scalars[tag] = ea.Scalars(tag)

    return scalars


def benchmark_results(benchmark_dir: str = "results/benchmark_jarvis"):
    """This function extracts the benchmark results from the tensorboard logs.

    Args:
        benchmark_dir (str, optional): a name of the directory where the tensorboard logs .

    Returns:
        pd.DataFrame: a dataframe with the benchmark results.
    """
    benchmark_dir = Path(benchmark_dir)
    data = []
    for p in benchmark_dir.glob("*/*"):
        scalar_data = extract_scalars(p)
        for tag, values in scalar_data.items():
            scalar_data = [v.value for v in values]
            if "test" in tag:
                data.append((str(p), tag, scalar_data))

    df = pd.DataFrame(data, columns=["path", "tag", "values"])
    df = df.sort_values(by=["path"])  # sort by path
    return df
