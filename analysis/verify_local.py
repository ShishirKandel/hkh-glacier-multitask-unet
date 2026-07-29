#!/usr/bin/env python3
"""verify_local.py -- re-verify the frozen local artefacts and write a transcript.

The Kaggle side of this project is evidenced by screenshots (Appendix B). The
local curate/quantise/shard/freeze stage produced files, not screens, so it is
evidenced instead by re-running its integrity checks and recording the output.
Sections 3 to 6 each recompute something from the artefacts on disk and compare
it against a value frozen before the first GPU run. Sections 1 and 2 identify the
machine and the source archive and are reported, not verified.

Where a frozen value could be rewritten alongside the thing it certifies, the
comparison is made against an INDEPENDENT copy: the evaluation-list hashes are
checked against analysis/shards/split_meta.json, which was uploaded to Kaggle
with the shards and never touched afterwards, not only against the split.json
that also stores the lists themselves.

Checks
  1  machine and interpreter the checks ran on
  2  source archive present, size, and the SHA-256 recorded at download time
  3  frozen evaluation lists: SHA-256 recomputed from split.json and compared
     against the eval_list_sha256 recorded at freeze time
  4  zone membership is disjoint, and no buffer or straddler patch leaked into
     an evaluation list
  5  shard census: patch counts and debris-pixel totals recomputed from the
     .npz shards and compared against the frozen split contract
  6  dequantised channel behaviour against the documented channel map, including
     the sign-flipped NDSI convention tested where it matters (over labelled ice
     rather than over a whole shard, whose mean depends on how snowy it is)

Writes verify_local.txt, which is reproduced verbatim at the end of the
Reproduction Guide appendix. Run it from the working tree that holds the source
archive and the built shards: a clone of the repository carries this script and
its recorded output, not the 73 GB of inputs it reads.

Usage:  python verify_local.py            (fast, ~2 min: reads every shard)
        python verify_local.py --hash-archive   (also re-hashes the 73 GB archive)
"""
import getpass
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis"
SHARDS = ANALYSIS / "shards"
ARCHIVE = ROOT / "hkh_patches.tar.gz"
OUT = HERE / "verify_local.txt"

NODATA = 65535
ZONES = ("train", "val", "test", "east")

lines = []
results = []


def say(s=""):
    print(s)
    lines.append(s)


def check(name, ok, detail=""):
    results.append(bool(ok))
    say("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  " + detail) if detail else ""))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    say("LOCAL ARTEFACT VERIFICATION")
    say("Sections 3 to 6 recompute values from the files on disk and compare them")
    say("against values frozen before the first GPU run. Sections 1 and 2 report")
    say("the machine and the source archive.")
    say()

    # ---- 1 machine ------------------------------------------------------
    say("1. MACHINE AND INTERPRETER  (reported, not verified)")
    say("  host           %s" % socket.gethostname())
    say("  user           %s" % getpass.getuser())
    say("  platform       %s %s (%s)" % (platform.system(), platform.release(),
                                         platform.machine()))
    say("  processor      %s" % (platform.processor() or "n/a"))
    say("  python         %s" % sys.version.split()[0])
    say("  numpy          %s" % np.__version__)
    try:
        import torch
        say("  torch          %s (cuda available: %s)"
            % (torch.__version__, torch.cuda.is_available()))
    except Exception as exc:                                  # noqa: BLE001
        say("  torch          not importable (%s)" % exc)
    say("  run at         %s UTC" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    say()

    # ---- 2 source archive ----------------------------------------------
    say("2. SOURCE ARCHIVE")
    recorded = (ANALYSIS / "archive_sha256.txt").read_text(encoding="utf-8").strip()
    recorded_hex = recorded.split("=")[-1].strip()
    if ARCHIVE.is_file():
        size = ARCHIVE.stat().st_size
        say("  file           %s" % ARCHIVE.name)
        say("  size           %s bytes (%.2f GB)" % (format(size, ","), size / 1e9))
        say("  recorded       %s" % recorded_hex)
        if "--hash-archive" in sys.argv:
            say("  recomputing SHA-256 over %.1f GB ..." % (size / 1e9))
            got = sha256_file(ARCHIVE)
            check("archive SHA-256 matches the value recorded at download",
                  got == recorded_hex, got)
        else:
            say("  (re-hashing 73 GB is skipped by default; run with --hash-archive)")
            try:                       # actually open it: is_file() proves nothing
                with open(ARCHIVE, "rb") as fh:
                    magic = fh.read(2)
                check("archive opens and starts with the gzip magic bytes",
                      magic == b"\x1f\x8b", magic.hex())
            except OSError as exc:     # noqa: BLE001
                check("archive opens", False, str(exc))
    else:
        check("archive present", False, "%s not found" % ARCHIVE)
    say()

    # ---- 3 frozen evaluation lists --------------------------------------
    say("3. FROZEN EVALUATION LISTS")
    split = json.loads((ANALYSIS / "split.json").read_text(encoding="utf-8"))
    # split.json stores BOTH the lists and their hashes, so on its own it can only
    # prove internal consistency. split_meta.json shipped to Kaggle with the shards
    # and carries the same hashes; agreeing with it is the check that has teeth.
    meta = json.loads((SHARDS / "split_meta.json").read_text(encoding="utf-8"))
    meta_hashes = meta.get("eval_list_sha256", {})
    for zone in ("val", "test", "east"):
        got = hashlib.sha256("\n".join(sorted(split[zone])).encode()).hexdigest()
        check("%-5s list SHA-256 unchanged since freeze" % zone,
              got == split["eval_list_sha256"][zone], got)
        check("%-5s hash also matches the copy shipped with the shards" % zone,
              bool(meta_hashes) and got == meta_hashes.get(zone),
              "no copy in split_meta.json" if not meta_hashes else "")
    say()

    # ---- 4 membership disjointness --------------------------------------
    say("4. ZONE MEMBERSHIP")
    sets = {z: set(split[z]) for z in ZONES}
    for a in ZONES:
        for b in ZONES:
            if a < b:
                check("%s and %s share no patch" % (a, b), not (sets[a] & sets[b]),
                      "" if not (sets[a] & sets[b]) else "%d shared" % len(sets[a] & sets[b]))
    excluded = set(split["buffer_removed"]) | set(split["dropped_straddlers"])
    evalset = sets["val"] | sets["test"] | sets["east"]
    check("no buffer or straddler patch reached an evaluation list",
          not (excluded & evalset))
    check("no buffer or straddler patch reached the training list",
          not (excluded & sets["train"]))
    say()

    # ---- 5 shard census --------------------------------------------------
    say("5. SHARD CENSUS RECOMPUTED FROM THE .npz FILES")
    say("  %-6s %9s %9s %9s %9s %11s %11s"
        % ("zone", "patches", "expected", "debris", "expected", "debris px", "expected"))
    spec = json.loads((SHARDS / "quant_spec.json").read_text(encoding="utf-8"))
    qmin = np.asarray(spec["qmin"], np.float32)
    scale = np.asarray(spec["scale"], np.float32)

    census_ok = True
    acc = {"ndsi_ice": [0.0, 0], "ndsi_bg": [0.0, 0], "ndsi_all": [0.0, 0],
           "elev": [np.inf, -np.inf], "slope": [np.inf, -np.inf]}
    for zone in ZONES:
        exp = split["counts"][zone]
        n = ndeb = 0
        deb_px = 0
        names = []
        for p in sorted((SHARDS / zone).glob("shard_s*.npz")):
            with np.load(p) as z:
                m, q = z["masks"], z["imgs"]
                names += [str(x) for x in z["names"]]
                n += m.shape[0]
                d = (m[..., 1] > 0)
                deb_px += int(d.sum())
                ndeb += int((d.reshape(d.shape[0], -1).any(axis=1)).sum())
                if zone == "train":                    # channel stats, train only
                    for ch, key in ((11, "ndsi"), (13, "elev"), (14, "slope")):
                        qc = q[..., ch]
                        ok = qc != NODATA
                        v = qmin[ch] + qc.astype(np.float32) * scale[ch]
                        if key == "ndsi":
                            ice, bg = (m[..., 0] > 0) & ok, (m.sum(-1) == 0) & ok
                            for tag, sel in (("ndsi_ice", ice), ("ndsi_bg", bg),
                                             ("ndsi_all", ok)):
                                acc[tag][0] += float(v[sel].sum())
                                acc[tag][1] += int(sel.sum())
                        elif ok.any():
                            acc[key][0] = min(acc[key][0], float(v[ok].min()))
                            acc[key][1] = max(acc[key][1], float(v[ok].max()))
        say("  %-6s %9d %9d %9d %9d %11s %11s"
            % (zone, n, exp["patches"], ndeb, exp["debris"],
               format(deb_px, ","), format(exp["debris_px"], ",")))
        ok = (n == exp["patches"] and ndeb == exp["debris"]
              and deb_px == exp["debris_px"] and set(names) == sets[zone])
        census_ok = census_ok and ok
    check("shard contents match the frozen split contract in every zone", census_ok)
    say()

    # ---- 6 dequantised channel behaviour ---------------------------------
    say("6. DEQUANTISED CHANNEL BEHAVIOUR (whole training zone)")
    mean = lambda k: acc[k][0] / max(acc[k][1], 1)
    say("  ch 11 mean over clean-ice labels   %+.4f  (n = %s)"
        % (mean("ndsi_ice"), format(acc["ndsi_ice"][1], ",")))
    say("  ch 11 mean over background         %+.4f  (n = %s)"
        % (mean("ndsi_bg"), format(acc["ndsi_bg"][1], ",")))
    say("  ch 11 mean over all valid pixels   %+.4f" % mean("ndsi_all"))
    check("ice is more negative than background in channel 11, confirming the "
          "sign-flipped NDSI convention", mean("ndsi_ice") < mean("ndsi_bg") - 0.1,
          "gap %.3f" % (mean("ndsi_bg") - mean("ndsi_ice")))
    stats = json.loads((SHARDS / "train_stats.json").read_text(encoding="utf-8"))
    check("recomputed channel-11 mean matches train_stats.json",
          abs(mean("ndsi_all") - stats["mean"][11]) < 1e-3,
          "%.4f vs %.4f" % (mean("ndsi_all"), stats["mean"][11]))
    say("  ch 13 elevation (m)                min %8.1f   max %8.1f"
        % (acc["elev"][0], acc["elev"][1]))
    say("  ch 14 slope (degrees)              min %8.1f   max %8.1f"
        % (acc["slope"][0], acc["slope"][1]))
    check("elevation within the SRTM range for the HKH (-500 to 9,000 m; the "
          "lower bound allows the documented small negative artifacts)",
          -500 <= acc["elev"][0] and acc["elev"][1] <= 9000)
    check("slope within 0 to 90 degrees",
          -0.01 <= acc["slope"][0] and acc["slope"][1] <= 90.01)
    say("  quantisation: %s, theoretical max abs error %.4g over the band channels,"
        % (spec["dtype"], max(spec["theoretical_max_abs_err"][:9])))
    say("  measured %.4g against %d float32 samples at build time."
        % (max(spec["measured_max_abs_err_on_samples"][:9]),
           spec["verified_against_n_float32_samples"]))
    say()

    say("RESULT: %d of %d checks passed in %.0f s."
        % (sum(results), len(results), time.time() - t0))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nverify_local: wrote %s" % OUT)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
