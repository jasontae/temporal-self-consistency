"""Rebuild the whole figure pack: every figure, the font check, and the zip.

    python3 figures/paper/scripts/make_all.py

Each fig_*.py reads only committed files under docs/orientation/results_sep10/
and writes <name>.pdf and <name>.png (300 dpi) into figures/paper/. style.save
runs the text audit (nothing under 6 pt, no overlapping text, nothing past the
page edge) on every figure. This script then checks with pdffonts (poppler)
that every font in every PDF is embedded and TrueType, and zips the pack.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ZIP = PACK / "figure-pack-2026-09-29.zip"
SCRIPTS = ["fig_threshold", "fig_staleness", "fig_crossover", "fig_vocab", "fig_inverse", "fig_frequency"]


def fonts_ok(pdf):
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
    bad = []
    for line in out:
        # pdffonts columns: name type encoding emb sub uni object ID
        if " yes " not in f" {line} " or "TrueType" not in line:
            bad.append(line.strip())
    return len(out), bad


def main():
    problems = 0
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, str(HERE / f"{s}.py")], cwd=HERE, capture_output=True, text=True)
        sys.stdout.write("".join(l + "\n" for l in r.stdout.splitlines() if l.startswith("QA")))
        if r.returncode:
            sys.stderr.write(r.stderr)
            raise SystemExit(f"{s} failed")
        problems += sum(1 for l in r.stdout.splitlines() if l.startswith("QA") and " problems" not in l)
    for pdf in sorted(PACK.glob("*.pdf")):
        n, bad = fonts_ok(pdf)
        print(f"fonts {pdf.name}: {n} fonts, {'all embedded TrueType' if not bad else 'NOT OK: ' + '; '.join(bad)}")
        problems += len(bad)
    files = sorted(PACK.glob("*.pdf")) + sorted(PACK.glob("*.png")) + sorted(PACK.glob("*.svg")) + [PACK / "README.md"]
    files += sorted(HERE.glob("*.py"))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                zi = zipfile.ZipInfo(str(f.relative_to(PACK)), date_time=(2026, 9, 29, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, f.read_bytes())
    print(f"zip {ZIP.name}: {len(files)} files")
    print(f"total QA problems: {problems}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
