"""Combinatorial demographic profile generator.

Instead of independently sampling each dimension, this script enumerates
*all* combinations of the chosen demographic attributes and generates a
fixed number of agents per profile.  This design is intentional: the goal
is NOT to mirror real population distributions but to run a controlled
synthetic experiment that tests whether agents with different profiles
produce different responses and whether similar profiles produce similar
responses.

With D dimensions the profile count is the Cartesian product of the value
sets.  Keep individual value sets small to avoid combinatorial explosion.

Profile count examples
----------------------
  2 × 2 × 2            =  8 profiles  →  125 agents / profile  (n = 1 000)
  2 × 2 × 2 × 3        = 24 profiles  →   42 agents / profile  (n = 1 008)
  2 × 2 × 2 × 2 × 3    = 48 profiles  →   21 agents / profile  (n = 1 008)

Usage
-----
    uv run python data/generate_demographics.py
    uv run python data/generate_demographics.py --agents-per-profile 50 --output data/demographics_50pp.csv
"""

import argparse
import itertools
import uuid
from pathlib import Path

import pandas as pd

# ── Demographic dimensions ─────────────────────────────────────────────────────
# Map dimension_name → list of possible values.
# The Cartesian product of all value lists defines the complete profile space.
#
# Rule of thumb: keep total profiles ≤ ~50 so each profile has enough agents
# for statistical comparison (≥ 20).  With the defaults below:
#   2 × 2 × 2 × 3 = 24 profiles  →  42 agents / profile  (n = 1 008)

DIMENSIONS: dict[str, list[str]] = {
    "gender": [
        "Male",
        "Female",
    ],
    "economic_status": [
        "Low income",
        "High income",
    ],
    "political_spectrum": [
        "Progressive",
        "Conservative",
    ],
    # Personality archetype — adds a 3× multiplier to profile count.
    # Each archetype describes how the persona *processes information*, which is
    # intentionally orthogonal to political orientation (an analytical
    # conservative differs from an empathetic conservative in image perception).
    # Comment this block out to reduce from 24 → 8 profiles.
    "personality": [
        "Analytical",   # data-driven, values logic and self-reliance
        "Empathetic",   # emotionally driven, values community and relationships
        "Pragmatic",    # outcome-focused, values stability and practical solutions
    ],
}


def build_profiles() -> pd.DataFrame:
    """Return a DataFrame with one row per unique demographic profile."""
    dim_names = list(DIMENSIONS.keys())
    dim_values = list(DIMENSIONS.values())
    rows = [
        dict(zip(dim_names, combo))
        for combo in itertools.product(*dim_values)
    ]
    profiles = pd.DataFrame(rows)
    profiles.insert(0, "profile_id", [str(uuid.uuid4()) for _ in range(len(profiles))])
    return profiles


def generate_demographics(agents_per_profile: int = 42) -> pd.DataFrame:
    """Replicate each profile `agents_per_profile` times.

    Args:
        agents_per_profile: Number of agents generated for every profile.

    Returns:
        DataFrame with columns: persona_id, profile_id, <dimension columns>.
        Rows are sorted by profile_id so agents of the same profile are
        contiguous, which simplifies downstream batched processing.
    """
    profiles = build_profiles()
    df = (
        pd.concat([profiles] * agents_per_profile, ignore_index=True)
        .sort_values("profile_id")
        .reset_index(drop=True)
    )
    df.insert(0, "persona_id", [str(uuid.uuid4()) for _ in range(len(df))])
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combinatorial demographic profiles for persona generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agents-per-profile",
        type=int,
        default=42,
        help="Number of agents generated per unique demographic profile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "baseline_distribution.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles = build_profiles()
    n_profiles = len(profiles)
    total = n_profiles * args.agents_per_profile

    value_counts = "  ×  ".join(str(len(v)) for v in DIMENSIONS.values())
    print(f"Dimensions:          {list(DIMENSIONS.keys())}")
    print(f"Profile space:       {value_counts}  =  {n_profiles} profiles")
    print(f"Agents per profile:  {args.agents_per_profile}")
    print(f"Total agents:        {total}")
    print()

    df = generate_demographics(agents_per_profile=args.agents_per_profile)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} rows → {args.output}")
    print("\nProfile table:")
    print(profiles.to_string(index=False))
    print("\nAgent counts per dimension value:")
    for col in DIMENSIONS:
        counts = df[col].value_counts().sort_index()
        print(f"\n  {col}:")
        for val, cnt in counts.items():
            print(f"    {val:<30} {cnt} agents")


if __name__ == "__main__":
    main()
