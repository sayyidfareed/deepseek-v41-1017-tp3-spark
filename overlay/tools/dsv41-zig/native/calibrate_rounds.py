# Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
"""Fit draft costs from the existing lane CLI's synthetic --profile packets.

This reads saved CPU/GPU timing records only. It never launches an engine or
changes its environment. Missing row widths are explicitly interpolated; no
value in that set is described as a direct measurement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Positive integer timing/count required")
    return value


def fit(paths, cap=48, minimum_count=10, repeats=3):
    if type(cap) is not int or cap != 48 or type(repeats) is not int or repeats < 3:
        raise ValueError("This candidate requires 48 rows and at least three repeats")
    if type(minimum_count) is not int or minimum_count < 10:
        raise ValueError("At least ten observations per width/depth are required")
    rounds, drafts = {}, {}
    seen_files = set()
    for path in paths:
        fingerprint = hashlib.sha256()
        packets = []
        with Path(path).open("rb") as stream:
            while line := stream.readline((2 << 20) + 1):
                if len(line) > 2 << 20:
                    raise ValueError("Oversized profile line")
                fingerprint.update(line)
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    continue  # Existing CLI progress/diagnostics are not profile packets.
                if isinstance(value, dict) and "calibration" in value:
                    packets.append(value["calibration"])
        digest = fingerprint.hexdigest()
        if digest in seen_files or len(packets) != 1:
            raise ValueError("Each run needs exactly one distinct profile file")
        seen_files.add(digest)
        packet = packets[0]
        if not isinstance(packet, dict):
            raise ValueError("Calibration packet must be an object")
        if type(packet.get("world")) is not int or type(packet.get("round_cap")) is not int:
            raise ValueError("Integer profile topology and row cap required")
        if packet["world"] != 3 or packet["round_cap"] != cap:
            raise ValueError("Profile topology or row cap differs")
        if any(type(packet.get(key)) is not int or packet[key] != 0
               for key in ("candidate_rounds", "candidate_fallback_rounds")):
            raise ValueError("Baseline calibration must use the full-logit sampler")
        for source, target, key in (("rounds", rounds, "rows"), ("solo_drafts", drafts, "steps")):
            seen_keys = set()
            for item in packet[source]:
                width = positive_int(item[key])
                if width in seen_keys or width > (cap if key == "rows" else 8):
                    raise ValueError("Repeated or out-of-range timing key")
                seen_keys.add(width)
                count = positive_int(item["count"])
                elapsed = positive_int(item["total_ns"])
                if count >= minimum_count:
                    target.setdefault(width, []).append(elapsed / count / 1e6)
    measured = {n: statistics.median(v) for n, v in rounds.items() if len(v) >= repeats}
    widths = sorted(measured)
    if len(widths) < 8 or widths[0] != 1 or widths[-1] != cap:
        raise ValueError("Insufficient measured coverage: need eight widths including 1 and 48")
    # Monotone envelope suppresses timer noise. Interpolate only between measured endpoints.
    anchors, last = {}, 0.0
    for width in widths:
        last = max(last, measured[width])
        anchors[width] = last
    curve = []
    for width in range(1, cap + 1):
        if width in anchors:
            curve.append(anchors[width])
            continue
        lower = max(n for n in widths if n < width)
        upper = min(n for n in widths if n > width)
        curve.append(anchors[lower] + (anchors[upper] - anchors[lower]) * (width - lower) / (upper - lower))
    solo = {n: statistics.median(v) for n, v in drafts.items() if len(v) >= repeats}
    if not solo:
        raise ValueError("No repeated solo draft-depth measurements")
    # A linear per-step model for the existing chooser, not an assertion that draft time is linear.
    draft_ms = statistics.median(value / steps for steps, value in solo.items())
    if any(not math.isfinite(v) or v <= 0 for v in curve + [draft_ms]):
        raise ValueError("Non-finite or nonpositive derived cost")
    return {
        "status": "FIT_READY_NOT_A_SPEED_OR_DEPLOYMENT_PASS",
        "profile_files": len(seen_files),
        "profile_sha256": sorted(seen_files),
        "independence_limit": "Distinct files alone do not prove independent runs; check the trial's process/configuration receipts.",
        "measured_width_ms": measured,
        "monotone_anchor_ms": anchors,
        "interpolated_widths": [n for n in range(1, cap + 1) if n not in measured],
        "solo_pass_ms_by_depth": solo,
        "draft_ms_model": "median measured solo pass time per requested draft step",
        "TF_DS_ROUND_MS": ",".join(format(v, ".8g") for v in curve),
        "TF_DS_DRAFT_MS": format(draft_ms, ".8g"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = fit(args.profiles)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
