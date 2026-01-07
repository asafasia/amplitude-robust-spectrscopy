import numpy as np
from pathlib import Path
from data_loader.parsing import parse_metadata
from data_loader.results import SpectroscopyResult


def load_directory(root: Path) -> list[SpectroscopyResult]:
    results = []
    for npz in root.rglob("*.npz"):
        meta = parse_metadata(npz)
        data = np.load(npz, allow_pickle=True)
        results.append(SpectroscopyResult(meta=meta, raw=data))

    return results


if __name__ == "__main__":

    from echospec.utils.config import PROJECT_PATH

    path = PROJECT_PATH / "data" / "data_old" / "27-1-25"
    data = load_directory(path)

    print(data)
    print(f"Loaded {len(data)} results")
    for result in data:
        print(result.meta)
