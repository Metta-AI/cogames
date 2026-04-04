#!/usr/bin/env python3
"""Generate a progress plot of autoresearch experiments across issues."""

import subprocess
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).parent.parent


def get_commit_dates():
    """Map short commit hash -> datetime from git log across all branches."""
    result = subprocess.run(
        ["git", "log", "--format=%h %aI", "--all"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    dates = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split(" ", 1)
        if len(parts) == 2:
            dates[parts[0]] = datetime.fromisoformat(parts[1])
    return dates


def parse_tsv(filepath, commit_dates):
    """Parse a TSV file and return (datetime, reward, status) tuples."""
    points = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            commit_raw = row.get("commit", "").strip()
            if not commit_raw:
                continue
            # Handle suffixed commits like "bcec881_a", notes, etc.
            base = commit_raw.split("_")[0]
            # Skip non-hash entries
            if not re.match(r'^[0-9a-f]{5,}$', base) and base not in ("baseline",):
                continue

            reward_str = row.get("mission_reward") or row.get("reward", "")
            try:
                reward = float(reward_str)
            except (ValueError, TypeError):
                continue

            # Rough normalization: scale by steps to per-1000-step equivalent
            steps_str = row.get("steps", "1000")
            try:
                steps = int(steps_str)
            except (ValueError, TypeError):
                steps = 1000
            if steps > 0 and steps != 1000:
                reward = reward * (1000 / steps)

            status = row.get("status", "").strip()

            # Look up date
            dt = commit_dates.get(base)
            if dt is None:
                # Try prefix match
                for k, v in commit_dates.items():
                    if k.startswith(base[:6]):
                        dt = v
                        break
            if dt is None:
                continue

            points.append((dt, reward, status))

    # Filter out points using different reward scales (total vs per-agent)
    points = [(d, r, s) for d, r, s in points if r <= 1.5]
    points.sort(key=lambda x: x[0])
    return points


def main():
    commit_dates = get_commit_dates()

    # All issue TSV files with display labels
    issues = [
        ("Mar 21 - 3-agent baseline", "docs/results_autoresearch_21_march.tsv"),
        ("Mar 22 - 3-agent alignment", "docs/results_autoresearch_22_march.tsv"),
        ("#9 cross-role policy", "docs/results_autoresearch_issue9_cross_role_policy.tsv"),
        ("#10 fixed-roles tuning", "docs/results_autoresearch_issue-10-fixed-roles-tuning.tsv"),
        ("#12 gear acquisition", "docs/results_autoresearch_issue12_gear_acquisition_reliability.tsv"),
        ("#16 hub depletion", "docs/results_autoresearch_issue-16-hub-depletion-awareness.tsv"),
        ("#16v2 hub depletion", "docs/results_autoresearch_issue-16-hub-depletion-awareness-v2.tsv"),
        ("#20 spatial partitioning", "docs/results_autoresearch_issue-20-coordinated-multi-agent-spatial-partitioning.tsv"),
        ("#21 intrinsic exploration", "docs/results_autoresearch_issue21_intrinsic_motivation_exploration.tsv"),
        ("#24 balanced mining", "docs/results_autoresearch-issue-24-balanced-mining.tsv"),
        ("#24 mining+makeheart", "docs/results_autoresearch_issue24_balanced_mining_makeheart.tsv"),
        ("#24 mining strategy", "docs/results_autoresearch_issue24_balanced_mining_strategy.tsv"),
        ("#25 8-agent scaling", "docs/results_autoresearch_issue-25-8agent-scaling-4a4m.tsv"),
    ]

    # Color palette
    palette = [
        "#58A6FF", "#3FB950", "#F0883E", "#BC8CFF", "#F778BA",
        "#79C0FF", "#56D364", "#D29922", "#DB61A2", "#FF7B72",
        "#7EE787", "#FFA657", "#D2A8FF",
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    plotted = 0
    for i, (label, tsv_rel) in enumerate(issues):
        tsv_path = REPO_ROOT / tsv_rel
        if not tsv_path.exists():
            continue
        points = parse_tsv(tsv_path, commit_dates)
        if len(points) < 2:
            continue

        color = palette[i % len(palette)]

        dates = [p[0] for p in points]
        rewards = [p[1] for p in points]

        # Running best from kept experiments
        best_dates, best_rewards = [], []
        running_best = -1
        for d, r, s in points:
            if s in ("keep", "completed") and r > running_best:
                running_best = r
                best_dates.append(d)
                best_rewards.append(r)

        # All experiments as faint dots
        ax.scatter(dates, rewards, color=color, alpha=0.12, s=12, zorder=2)

        # Running-best step line
        if len(best_dates) >= 2:
            step_dates, step_rewards = [], []
            for j, (d, r) in enumerate(zip(best_dates, best_rewards)):
                if j > 0:
                    step_dates.append(d)
                    step_rewards.append(best_rewards[j - 1])
                step_dates.append(d)
                step_rewards.append(r)

            ax.plot(step_dates, step_rewards, color=color, linewidth=2.2,
                    label=f"{label} ({best_rewards[-1]:.2f})", zorder=3)
            ax.scatter(best_dates, best_rewards, color=color, s=40,
                       edgecolors="white", linewidth=0.7, zorder=4)
            plotted += 1
        elif best_dates:
            ax.scatter(best_dates, best_rewards, color=color, s=60,
                       edgecolors="white", linewidth=0.7, zorder=4,
                       label=f"{label} ({best_rewards[-1]:.2f})")
            plotted += 1

    # Styling
    ax.set_xlabel("Date", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Mission Reward", color="#c9d1d9", fontsize=12)
    ax.set_title("CoGames Autoresearch: Autonomous Agent Improvement Over Time\n"
                 "Each line = one research issue, dots = all experiments, line = running best",
                 color="white", fontsize=14, fontweight="bold", pad=15)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.08, color="white")

    ax.set_ylim(bottom=0)

    legend = ax.legend(loc="upper left", fontsize=8.5, ncol=2,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="white")

    plt.tight_layout()
    out_path = REPO_ROOT / "docs/autoresearch_progress.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved plot to {out_path}")
    print(f"Plotted {plotted} issues")


if __name__ == "__main__":
    main()
