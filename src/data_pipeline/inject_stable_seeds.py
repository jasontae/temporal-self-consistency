"""
Stable Seed Example Injector
============================
Fixes the [CONFIDENT] underrepresentation discovered in the training data.

The deploy-offset augmentation that fixed [UNKNOWN] starvation had a side
effect: [CONFIDENT] dropped to ~40 records (0.06%) and immutable facts to 40.
A model trained on this will rarely see "be confident about a stable fact"
and will tend to over-hedge — which the adversarial-stable stress test is
designed to catch.

This script injects curated immutable / stable-fact examples labeled
[CONFIDENT], each replicated across the same deployment-date offsets used in
the main pipeline (so they're not trivially distinguishable by offset).

Usage:
    python inject_stable_seeds.py \\
        --train temporal_delta_train.jsonl \\
        --output temporal_delta_train_balanced.jsonl \\
        --target-confident-pct 10
"""
import argparse
import json

# Curated immutable facts. Physical constants, math, historical dates,
# authorship, geography that does not change on any relevant timescale.
STABLE_SEEDS = [
    ("What is the speed of light?", "299,792,458 metres per second"),
    ("What is the boiling point of water at standard pressure?", "100 degrees Celsius"),
    ("What is the freezing point of water at standard pressure?", "0 degrees Celsius"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the chemical formula of water?", "H2O"),
    ("What is the chemical formula of carbon dioxide?", "CO2"),
    ("What is the speed of sound in air at sea level?", "approximately 343 metres per second"),
    ("What is the gravitational acceleration on Earth?", "9.81 metres per second squared"),
    ("What is absolute zero in Celsius?", "-273.15 degrees Celsius"),
    ("What is the value of pi to two decimal places?", "3.14"),
    ("What is the value of e to two decimal places?", "2.72"),
    ("How many degrees are in a triangle?", "180"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the square root of 144?", "12"),
    ("In what year did World War 2 end?", "1945"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did Apollo 11 land on the moon?", "1969"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("In what year was the Declaration of Independence signed?", "1776"),
    ("In what year did the Titanic sink?", "1912"),
    ("In what year did the French Revolution begin?", "1789"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who composed the Ninth Symphony?", "Ludwig van Beethoven"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("What is the tallest mountain on Earth above sea level?", "Mount Everest"),
    ("What is the longest river in South America?", "Amazon"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the closest star to Earth?", "the Sun"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the powerhouse of the cell?", "the mitochondria"),
    ("How many chromosomes does a typical human cell have?", "46"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("How many moons does Mars have?", "2"),
    ("On which continent is Egypt located?", "Africa"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--output", default="temporal_delta_train_balanced.jsonl")
    ap.add_argument("--target-confident-pct", type=float, default=10.0)
    ap.add_argument("--offsets", default="0,6,12,18,24",
                    help="deployment-date offsets to replicate seeds across")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.train)]
    n_total = len(records)
    n_confident = sum(1 for r in records
                      if r.get("hedge_token", r.get("hedge")) == "[CONFIDENT]")

    offsets = [int(x) for x in args.offsets.split(",")]

    # How many [CONFIDENT] records do we need to hit the target percentage?
    # target_pct = (n_confident + added) / (n_total + added)
    t = args.target_confident_pct / 100.0
    needed = max(0, int((t * n_total - n_confident) / (1 - t)))

    print(f"Current [CONFIDENT]: {n_confident}/{n_total} "
          f"({100*n_confident/n_total:.2f}%)")
    print(f"Target: {args.target_confident_pct}% -> need to add ~{needed} records")

    # Build seed records, cycling through the curated list across offsets
    new_records = []
    seed_i = 0
    while len(new_records) < needed:
        q, a = STABLE_SEEDS[seed_i % len(STABLE_SEEDS)]
        offset = offsets[(seed_i // len(STABLE_SEEDS)) % len(offsets)]
        new_records.append({
            "question": q,
            "answer": a,
            "hedge_token": "[CONFIDENT]",
            "formatted_output": f"Answer: {a} / Hedge: [CONFIDENT]",
            "volatility": "immutable",
            "validity_start": None,
            "validity_end": None,
            "deploy_offset_months": offset,
            "effective_query_year": 2024 + offset / 12.0,
            "months_past_validity_end": 0,
            "entity_id": None,
            "entity_label": None,
            "relation": "immutable_seed",
            "source": "stable_seed_injection",
        })
        seed_i += 1

    combined = records + new_records
    # Shuffle so seeds aren't all at the end
    import random
    random.seed(42)
    random.shuffle(combined)

    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_confident = n_confident + len(new_records)
    print(f"Added {len(new_records)} stable seed records")
    print(f"New [CONFIDENT]: {final_confident}/{len(combined)} "
          f"({100*final_confident/len(combined):.2f}%)")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
