#!/usr/bin/env python3
import os, sys, json, shutil, subprocess, tempfile, time

VENV_PYTHON = "/Users/AnanyaTyagi/Desktop/ScannedPdfFormatter/.venv/bin/python"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DHC_BASE    = os.path.expanduser("~/Desktop/DHCProject/DHC TEST CASES")
OUTPUT_ROOT = os.path.expanduser("~/Desktop/ML_Stats_Tagged")

BATCHES = {
    "well_encoded": ["Batch1", "Batch2", "Batch3", "Batch4", "Batch5", "Batch6", "Batch7"],
    "flyers":       ["Flyers1", "Flyers2", "Flyers3"],
}

def run_pipeline(pdf_path, work_dir):
    """Run TagMyPDF2.py on a single PDF inside work_dir. Returns True on success."""
    src = os.path.join(work_dir, "temp.pdf")
    shutil.copy2(pdf_path, src)
    tagmypdf = os.path.join(SCRIPT_DIR, "TagMyPDF2.py")
    result = subprocess.run(
        [VENV_PYTHON, "-u", tagmypdf, "temp.pdf"],
        cwd=work_dir,
        capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0

def read_stats(work_dir):
    """Read fuse_stats.json written by FuseLinesFixed_fixed.py."""
    stats_path = os.path.join(work_dir, "fuse_stats.json")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path) as f:
        return json.load(f)

def make_work_dir():
    d = tempfile.mkdtemp(prefix="tagpdf_")
    for sub in ("fused_tags", "layout_out", "lines_out", "structures", "debug"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d

def fmt(count, total):
    pct = count / total * 100 if total else 0
    return f"{count:6d}  ({pct:5.1f}%)"

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    groups = {}   # group_name -> {ml, heuristic, artifact, total, ok, fail}
    all_pdf_count = 0

    for group_name, batch_list in BATCHES.items():
        groups[group_name] = dict(ml=0, heuristic=0, artifact=0, total=0, ok=0, fail=0,
                                  source_breakdown={}, tag_types={})

        for batch in batch_list:
            batch_dir = os.path.join(DHC_BASE, batch)
            pdfs = sorted(
                p for p in (os.path.join(batch_dir, f) for f in os.listdir(batch_dir))
                if p.lower().endswith(".pdf")
            )
            print(f"\n{'='*60}")
            print(f"  {batch} ({len(pdfs)} PDFs)")
            print(f"{'='*60}")

            for pdf_path in pdfs:
                name = os.path.basename(pdf_path)
                all_pdf_count += 1
                work_dir = make_work_dir()
                t0 = time.time()
                try:
                    ok = run_pipeline(pdf_path, work_dir)
                    stats = read_stats(work_dir)
                    elapsed = time.time() - t0

                    if ok and stats:
                        g = groups[group_name]
                        g["ml"]        += stats["ml_tags"]
                        g["heuristic"] += stats["heuristic_tags"]
                        g["artifact"]  += stats["artifact_tags"]
                        g["total"]     += stats["total_tags"]
                        g["ok"]        += 1
                        for k, v in stats.get("source_breakdown", {}).items():
                            g["source_breakdown"][k] = g["source_breakdown"].get(k, 0) + v
                        for k, v in stats.get("tag_types", {}).items():
                            g["tag_types"][k] = g["tag_types"].get(k, 0) + v

                        # Copy tagged PDF to output folder
                        out_pdf = os.path.join(work_dir, "output.pdf")
                        if os.path.exists(out_pdf):
                            dest = os.path.join(OUTPUT_ROOT, f"{batch}__{name}")
                            shutil.copy2(out_pdf, dest)

                        print(f"  OK  {name[:55]:<55} {elapsed:5.1f}s  "
                              f"ML={stats['ml_tags']} H={stats['heuristic_tags']}")
                    else:
                        groups[group_name]["fail"] += 1
                        print(f"  FAIL {name[:55]}")
                finally:
                    shutil.rmtree(work_dir, ignore_errors=True)

    # ── Final summary ──────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ML vs HEURISTICS BREAKDOWN BY GROUP")
    print("="*70)
    print(f"  {'Group':<20} {'Total tags':>12} {'ML':>14} {'Heuristic':>14} {'Artifact':>12}")
    print("-"*70)

    grand = dict(ml=0, heuristic=0, artifact=0, total=0)
    for group_name, g in groups.items():
        label = "Well-encoded (Batch1-7)" if group_name == "well_encoded" else "Flyers (Flyers1-3)"
        t = g["total"]
        print(f"  {label:<20} {t:>12,}  {fmt(g['ml'],t)}  {fmt(g['heuristic'],t)}  {fmt(g['artifact'],t)}")
        for k in ("ml", "heuristic", "artifact", "total"):
            grand[k] += g[k]
        print(f"  {'':20}  PDFs: {g['ok']} OK, {g['fail']} failed")

    print("-"*70)
    t = grand["total"]
    print(f"  {'Overall':<20} {t:>12,}  {fmt(grand['ml'],t)}  {fmt(grand['heuristic'],t)}  {fmt(grand['artifact'],t)}")
    print("="*70)

    print("\n\nSource breakdown (overall):")
    combined = {}
    for g in groups.values():
        for k, v in g["source_breakdown"].items():
            combined[k] = combined.get(k, 0) + v
    for src, count in sorted(combined.items(), key=lambda x: x[1], reverse=True):
        print(f"  {src:<40} {count:6d}  ({count/t*100:.1f}%)" if t else f"  {src}")

    # Save JSON results
    results_path = os.path.join(OUTPUT_ROOT, "_ml_stats_summary.json")
    with open(results_path, "w") as f:
        json.dump({"groups": groups, "grand": grand}, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print(f"Tagged PDFs:   {OUTPUT_ROOT}/")

if __name__ == "__main__":
    main()
