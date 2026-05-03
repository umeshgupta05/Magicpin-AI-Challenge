#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from composer import compose


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(root: Path) -> tuple[dict, dict, dict, dict]:
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    if (root / "categories").exists():
        for path in (root / "categories").glob("*.json"):
            data = load_json(path)
            categories[data["slug"]] = data
    else:
        for path in (root / "dataset" / "categories").glob("*.json"):
            data = load_json(path)
            categories[data["slug"]] = data

    if (root / "merchants").exists():
        for path in (root / "merchants").glob("*.json"):
            data = load_json(path)
            merchants[data["merchant_id"]] = data
    else:
        for item in load_json(root / "dataset" / "merchants_seed.json")["merchants"]:
            merchants[item["merchant_id"]] = item

    if (root / "customers").exists():
        for path in (root / "customers").glob("*.json"):
            data = load_json(path)
            customers[data["customer_id"]] = data
    else:
        for item in load_json(root / "dataset" / "customers_seed.json")["customers"]:
            customers[item["customer_id"]] = item

    if (root / "triggers").exists():
        for path in (root / "triggers").glob("*.json"):
            data = load_json(path)
            triggers[data["id"]] = data
    else:
        for item in load_json(root / "dataset" / "triggers_seed.json")["triggers"]:
            triggers[item["id"]] = item

    return categories, merchants, customers, triggers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="expanded", help="Expanded dataset directory")
    parser.add_argument("--out", default="submission.jsonl")
    args = parser.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        raise SystemExit("Run: python dataset/generate_dataset.py --seed-dir dataset --out expanded")

    categories, merchants, customers, triggers = load_dataset(data_root)
    pair_path = data_root / "test_pairs.json"
    if not pair_path.exists():
        raise SystemExit(f"Missing {pair_path}")
    pairs = load_json(pair_path)["pairs"]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            merchant = merchants[pair["merchant_id"]]
            trigger = triggers[pair["trigger_id"]]
            category = categories[merchant["category_slug"]]
            customer = customers.get(pair.get("customer_id")) if pair.get("customer_id") else None
            result = compose(category, merchant, trigger, customer)
            result = {"test_id": pair["test_id"], **result}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} lines to {out_path}")


if __name__ == "__main__":
    main()
