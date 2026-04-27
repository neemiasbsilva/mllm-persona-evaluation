#!/usr/bin/env python3
"""
Kruskal-Wallis Test for Cross-Profile Sentiment Variation

This script performs a Kruskal-Wallis H-test to determine whether
demographic profiles produce statistically distinct sentiment distributions.

Usage:
    python scripts/kruskal_wallis_test.py [--input <path>] [--output <path>]

Example:
    python scripts/kruskal_wallis_test.py --input outputs/annotations_baseline.jsonl
    python scripts/kruskal_wallis_test.py  # uses defaults
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import kruskal
except ImportError:
    print("ERROR: scipy not installed. Install with: pip install scipy")
    raise SystemExit(1)


SENTIMENT_SCORE_MAP = {
    'Negative': -2,
    'SlightlyNegative': -1,
    'Neutral': 0,
    'SlightlyPositive': 1,
    'Positive': 2,
}


def load_annotations(input_path):
    """Load annotation data from JSONL file."""
    annotations = []
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                annotations.append(obj)
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {line_num} failed to parse: {e}")
                continue
    return annotations


def group_by_profile(annotations):
    """Group sentiment scores by demographic profile."""
    by_profile = defaultdict(list)
    
    for obj in annotations:
        demographics = obj.get('raw_demographics', {})
        sentiment = obj.get('predicted_sentiment')
        
        if not demographics or sentiment not in SENTIMENT_SCORE_MAP:
            continue
        
        # Create composite profile key
        profile_key = (
            f"{demographics.get('gender', '?')}|"
            f"{demographics.get('economic_status', '?')}|"
            f"{demographics.get('political_spectrum', '?')}|"
            f"{demographics.get('personality', '?')}"
        )
        
        score = SENTIMENT_SCORE_MAP[sentiment]
        by_profile[profile_key].append(score)
    
    return by_profile


def compute_kruskal_wallis(by_profile):
    """Compute Kruskal-Wallis test and effect size."""
    samples = [v for v in by_profile.values() if len(v) > 0]
    
    if len(samples) < 2:
        raise ValueError("Need at least 2 non-empty groups")
    
    H, p = kruskal(*samples)
    
    # Epsilon-squared: (H - k + 1) / (N - k)
    N = sum(len(v) for v in samples)
    k = len(samples)
    eps2 = (H - k + 1) / (N - k) if (N - k) > 0 else 0
    
    return H, p, eps2, k, N


def report_results(by_profile, H, p, eps2, k, N):
    """Print formatted results."""
    means = sorted(
        ((sum(v) / len(v), prof, len(v)) for prof, v in by_profile.items()),
        key=lambda x: x[0]
    )
    
    print("=" * 80)
    print("KRUSKAL-WALLIS TEST: Cross-Profile Sentiment Variation")
    print("=" * 80)
    print()
    print(f"Number of profiles (k):              {k}")
    print(f"Total annotations (N):               {N}")
    print(f"Average annotations per profile:     {N / k:.1f}")
    print()
    print("TEST RESULTS:")
    print(f"  H-statistic:                       {H:.4f}")
    print(f"  p-value:                           {p:.2e}")
    print(f"  Effect size (ε²):                  {eps2:.6f}")
    print()
    
    if p < 0.001:
        sig = "*** (highly significant)"
    elif p < 0.01:
        sig = "** (very significant)"
    elif p < 0.05:
        sig = "* (significant)"
    else:
        sig = "(not significant)"
    
    print(f"Interpretation: p < 0.05? {p < 0.05} {sig}")
    print()
    
    print("PROFILE EXTREMES (sorted by mean sentiment score):")
    print(f"  MIN:  {means[0][1]:50s} μ = {means[0][0]:7.4f}  (n={means[0][2]:4d})")
    print(f"  MAX:  {means[-1][1]:50s} μ = {means[-1][0]:7.4f}  (n={means[-1][2]:4d})")
    print(f"  RANGE:                                             Δμ = {means[-1][0] - means[0][0]:.4f}")
    print()
    
    print("LaTeX for Paper:")
    print(f"  $H({k-1})={H:.2f}$, $p={p:.2e}$, $\\varepsilon^2={eps2:.4f}$")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Kruskal-Wallis test for cross-profile sentiment variation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='outputs/annotations_baseline.jsonl',
        help='Path to annotations JSONL file (default: outputs/annotations_baseline.jsonl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional: save results to CSV file'
    )
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        raise SystemExit(1)
    
    print(f"Loading annotations from: {input_path}")
    annotations = load_annotations(input_path)
    print(f"Loaded {len(annotations)} annotations")
    print()
    
    print("Grouping by demographic profile...")
    by_profile = group_by_profile(annotations)
    print(f"Found {len(by_profile)} profiles")
    print()
    
    print("Computing Kruskal-Wallis test...")
    H, p, eps2, k, N = compute_kruskal_wallis(by_profile)
    
    report_results(by_profile, H, p, eps2, k, N)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("profile,n_annotations,mean_sentiment\n")
            for prof, scores in sorted(by_profile.items()):
                mean = sum(scores) / len(scores)
                f.write(f'"{prof}",{len(scores)},{mean:.6f}\n')
        print(f"Profile summary saved to: {output_path}")


if __name__ == '__main__':
    main()
