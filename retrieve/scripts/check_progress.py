"""
Check evaluation progress for retrieve output directory(s).

Usage:
    python retrieve/scripts/check_progress.py <output_dir>
    python retrieve/scripts/check_progress.py <dir1> <dir2> <dir3>
    python retrieve/scripts/check_progress.py <output_dir> --data data/longvideobench/video_info.meta.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# LVBench category short names
CATEGORY_MAP = {
    "entity recognition": "ER",
    "event understanding": "EU",
    "key information retrieval": "KIR",
    "temporal grounding": "TG",
    "reasoning": "Reason",
    "summarization": "Summ",
}
CATEGORY_ORDER = ["ER", "EU", "KIR", "Reason", "TG", "Summ"]


def is_lvbench(data_file: str) -> bool:
    return "LVBench" in str(data_file)


def load_data(data_file: str) -> dict:
    video_data = {}
    with open(data_file) as f:
        for line in f:
            d = json.loads(line)
            video_data[d["key"]] = {
                "total": len(d.get("qa", [])),
                "type": d.get("type", "unknown"),
                "question_types": [qt for qa in d.get("qa", []) for qt in qa.get("question_type", [])],
                "uid_types": {str(qa["uid"]): qa.get("question_type", []) for qa in d.get("qa", [])},
            }
    return video_data


def scan_output_dir(output_dir: Path, per_video: dict, uid_results: dict):
    if not output_dir.exists():
        print(f"  [WARN] Directory not found: {output_dir}")
        return

    for video_dir in sorted(output_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        summary_file = video_dir / "summary.json"
        if not summary_file.exists():
            q_correct = 0
            q_total = 0
            for fr_file in video_dir.glob("q_*/final_result.json"):
                try:
                    with open(fr_file) as f:
                        d = json.load(f)
                    q_total += 1
                    is_correct = d.get("correct", False)
                    if is_correct:
                        q_correct += 1
                    uid = fr_file.parent.name[2:]
                    uid_results[uid] = is_correct
                except:
                    pass
            if q_total == 0:
                continue
            v_name = video_dir.name.split("_2")[0]
            _merge_video(per_video, v_name, q_total, q_correct)
        else:
            with open(summary_file) as f:
                d = json.load(f)
            v_total = d.get("total", 0)
            v_correct = d.get("correct", 0)
            v_name = d.get("video", video_dir.name.split("_2")[0])
            _merge_video(per_video, v_name, v_total, v_correct)
            for r in d.get("results", []):
                uid_results[str(r["uid"])] = r.get("correct", False)


def _merge_video(per_video: dict, v_name: str, answered: int, correct: int):
    if v_name in per_video:
        per_video[v_name]["answered"] += answered
        per_video[v_name]["correct"] += correct
    else:
        per_video[v_name] = {"answered": answered, "correct": correct}


def check_progress(output_dirs: list, data_file: str):
    output_dirs = [Path(d) for d in output_dirs]
    for d in output_dirs:
        if not d.exists():
            print(f"Directory not found: {d}")
            return

    video_data = load_data(data_file)
    video_totals = {k: v["total"] for k, v in video_data.items()}
    is_lvb = is_lvbench(data_file)

    per_video = {}
    uid_results = {}

    for output_dir in output_dirs:
        scan_output_dir(output_dir, per_video, uid_results)

    completed = 0
    correct = 0
    total_answered = 0
    for v_name, stats in per_video.items():
        total_answered += stats["answered"]
        correct += stats["correct"]
        if v_name in video_totals and stats["answered"] >= video_totals[v_name]:
            completed += 1

    total_videos = len(video_totals)
    total_questions = sum(video_totals.values())
    acc = correct / total_answered if total_answered else 0
    progress = total_answered / total_questions * 100 if total_questions else 0

    dirs_str = str(output_dirs[0]) if len(output_dirs) == 1 else f"{len(output_dirs)} dirs: " + ", ".join(str(d) for d in output_dirs)
    print(f"{'='*55}")
    print(f"  Evaluation Progress: {dirs_str}")
    print(f"  Data: {data_file}")
    print(f"{'='*55}")
    print(f"  Videos:     {completed} / {total_videos} completed")
    print(f"  Questions:  {total_answered} / {total_questions} answered ({progress:.1f}%)")
    print(f"  Correct:    {correct}")
    print(f"  Accuracy:   {acc:.2%}")
    print(f"{'='*55}")

    if is_lvb:
        all_uid_types = {}
        for v_info in video_data.values():
            all_uid_types.update(v_info["uid_types"])

        cat_stats = {short: {"correct": 0, "total": 0} for short in CATEGORY_ORDER}
        for uid, is_correct in uid_results.items():
            types = all_uid_types.get(uid, [])
            if not types:
                continue
            for t in types:
                short = CATEGORY_MAP.get(t)
                if short and short in cat_stats:
                    cat_stats[short]["total"] += 1
                    if is_correct:
                        cat_stats[short]["correct"] += 1

        header = f"\n  {'Category':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}"
        print(header)
        print(f"  {'-'*38}")
        for cat in CATEGORY_ORDER:
            s = cat_stats[cat]
            c_acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"  {cat:<12} {s['correct']:>8} {s['total']:>8} {c_acc:>7.2%}")
        print(f"  {'-'*38}")
        print(f"  {'Avg':<12} {correct:>8} {total_answered:>8} {acc:>7.2%}")
    else:
        type_stats = defaultdict(lambda: {"correct": 0, "total": 0, "videos": 0})
        for v_name, stats in per_video.items():
            v_info = video_data.get(v_name)
            if not v_info:
                continue
            v_type = v_info["type"]
            type_stats[v_type]["correct"] += stats["correct"]
            type_stats[v_type]["total"] += stats["answered"]
            type_stats[v_type]["videos"] += 1

        if len(type_stats) > 1:
            header = f"\n  {'Type':<15} {'Videos':>6} {'Correct':>8} {'Total':>6} {'Acc':>8}"
            print(header)
            print(f"  {'-'*45}")
            for vtype in sorted(type_stats.keys()):
                s = type_stats[vtype]
                t_acc = s["correct"] / s["total"] if s["total"] else 0
                print(f"  {vtype:<15} {s['videos']:>6} {s['correct']:>8} {s['total']:>6} {t_acc:>7.2%}")

    incomplete = []
    for v, q_total in video_totals.items():
        if v in per_video:
            answered = per_video[v]["answered"]
            if answered < q_total:
                incomplete.append((v, answered, q_total))
        else:
            incomplete.append((v, 0, q_total))

    if incomplete:
        print(f"\n  Incomplete videos ({len(incomplete)}):")
        for v, answered, total in sorted(incomplete, key=lambda x: x[1]/x[2] if x[2] else 0):
            print(f"    {v}: {answered}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check evaluation progress")
    parser.add_argument("output_dirs", nargs="+", help="One or more retrieve output directories")
    parser.add_argument("--data", default="data/LVBench/video_info.meta.jsonl",
                        help="Question data file (default: LVBench)")
    args = parser.parse_args()
    check_progress(args.output_dirs, args.data)
