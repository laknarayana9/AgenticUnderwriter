"""Synthesize the extraction fine-tuning dataset.

Builds (free-text producer note -> canonical HO3 JSON) pairs in the OpenAI chat
fine-tuning format, with deliberate lexical variety and random field omission so
the model learns to normalize phrasing AND to abstain (null) on unstated fields.

Deterministic: a fixed seed makes the dataset reproducible. Output is written as
JSONL train/holdout splits; the data dir is gitignored, so regenerate with:

    python -m finetune.generate_dataset --train 3000 --holdout 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.schema import (  # noqa: E402
    DWELLING_VALUES,
    OCCUPANCY_VALUES,
    SYSTEM_PROMPT,
    empty_target,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

NAMES = [
    "Avery Chen", "Blake Rivera", "Casey Patel", "Dana Morgan", "Elliot Stone",
    "Finley Brooks", "Gray Harper", "Hayden Lee", "Jordan Kim", "Logan Price",
    "Morgan Fox", "Parker Young", "Quinn Adams", "Reese Nelson", "Sawyer Cruz",
    "Samantha Goldberg", "Robert Fitzgerald", "Maria Delgado-Torres", "Wei-Lin Chen",
]
STREETS = ["Market St", "Oak Ave", "Cedar Ln", "Ridge Rd", "Coastal Hwy", "Grove Ave", "Hill St"]
CITIES = ["Palo Alto", "San Diego", "Oakland", "Sacramento", "Long Beach", "Fresno", "Napa"]

_OCCUPANCY_PHRASES = {
    "owner_occupied_primary": ["their primary residence", "owner-occupied as a primary home", "where the owners live full time"],
    "owner_occupied_secondary": ["a secondary/seasonal home", "the owner's vacation property", "a second home"],
    "tenant_occupied": ["a rental occupied by tenants", "tenant-occupied", "leased to renters"],
    "vacant": ["currently vacant", "sitting empty right now", "unoccupied"],
}
_DWELLING_PHRASES = {
    "single_family": ["single-family home", "detached house"],
    "condo": ["condo", "condominium unit"],
    "townhouse": ["townhouse", "townhome"],
    "row_house": ["row house"],
    "commercial": ["commercial building", "commercial property"],
}


def _sample_record(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    """Return (note_text, gold_target). Each field is independently kept/dropped;
    a dropped field is absent from the note and null in the gold."""
    name = rng.choice(NAMES)
    address = f"{rng.randint(100, 1999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}, CA {rng.randint(94000, 96999)}"
    occupancy = rng.choice(OCCUPANCY_VALUES)
    dwelling = rng.choice(DWELLING_VALUES)
    year_built = rng.randint(1900, 2025)
    roof_age = rng.randint(0, 30)
    coverage_a = rng.choice([250, 300, 350, 400, 450, 500, 600, 750]) * 1000
    deductible = rng.choice([500, 1000, 1500, 2500, 5000])

    gold = empty_target()
    clauses: List[str] = []

    # Keep name + address most of the time (they anchor the note); others vary.
    def keep(prob: float) -> bool:
        return rng.random() < prob

    if keep(0.95):
        gold["applicant_name"] = name
        clauses.append(rng.choice([f"{name} is applying for a homeowners policy", f"Applicant: {name}"]))
    else:
        clauses.append("New homeowners application")

    if keep(0.92):
        gold["property_address"] = address
        clauses.append(rng.choice([f"on the property at {address}", f"for {address}"]))

    if keep(0.7):
        gold["dwelling_type"] = dwelling
        clauses.append(f"It's a {rng.choice(_DWELLING_PHRASES[dwelling])}")

    if keep(0.65):
        gold["occupancy"] = occupancy
        clauses.append(rng.choice(_OCCUPANCY_PHRASES[occupancy]).capitalize())

    if keep(0.7):
        gold["year_built"] = year_built
        clauses.append(rng.choice([f"built in {year_built}", f"a {year_built} build", f"constructed back in {year_built}"]))

    if keep(0.7):
        gold["roof_age_years"] = roof_age
        clauses.append(rng.choice([f"the roof is about {roof_age} years old", f"re-roofed {roof_age} years ago", f"roof age {roof_age}"]))

    if keep(0.6):
        gold["coverage_a"] = coverage_a
        clauses.append(rng.choice([f"they want ${coverage_a:,} of dwelling coverage", f"Coverage A of {coverage_a // 1000}k"]))

    if keep(0.6):
        gold["deductible"] = deductible
        clauses.append(rng.choice([f"with a ${deductible:,} deductible", f"deductible {deductible}"]))

    rng.shuffle(clauses)
    note = ". ".join(clauses) + "."
    return note, gold


def _to_chat_example(note: str, gold: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": note},
            {"role": "assistant", "content": json.dumps(gold)},
        ]
    }


def build_examples(count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [_to_chat_example(*_sample_record(rng)) for _ in range(count)]


def write_jsonl(path: Path, examples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(ex, separators=(",", ":")) for ex in examples) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the extraction fine-tuning dataset.")
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--holdout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    out = Path(args.out_dir)
    train = build_examples(args.train, args.seed)
    holdout = build_examples(args.holdout, args.seed + 1)  # disjoint seed stream
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "holdout.jsonl", holdout)
    print(f"Wrote {len(train)} train and {len(holdout)} holdout examples to {out}/")


if __name__ == "__main__":
    main()
