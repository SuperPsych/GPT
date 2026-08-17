import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO_ID = "HuggingFaceTB/smol-smoltalk"
PARQUET_NAMES = [f"train-0000{i}-of-00004.parquet" for i in range(4)]
PAGE_SIZE = 500


def download_parquets(data_dir: Path, force: bool) -> list[Path]:
    paths = []
    for name in PARQUET_NAMES:
        dest = data_dir / name
        if dest.exists() and not force:
            print(f"{dest} already exists, skipping download")
            paths.append(dest)
            continue
        print(f"downloading {name} from {REPO_ID}...")
        cached_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"data/{name}")
        shutil.copy2(cached_path, dest)
        paths.append(dest)
    return paths


def convert_to_pages(parquet_paths: list[Path], out_path: Path, page_size: int) -> None:
    total_rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in parquet_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=["messages"], batch_size=page_size):
                messages_col = batch.column("messages").to_pylist()
                rows = [{"row": {"messages": m}} for m in messages_col]
                out.write(json.dumps({"rows": rows}) + "\n")
                total_rows += len(rows)
                if total_rows % (page_size * 50) == 0:
                    print(f"{total_rows} conversations written...")
    print(f"{total_rows} conversations written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = download_parquets(data_dir, args.force)
    out_path = data_dir / "smoltalk_train_pages.jsonl"
    convert_to_pages(parquet_paths, out_path, args.page_size)


if __name__ == "__main__":
    main()
