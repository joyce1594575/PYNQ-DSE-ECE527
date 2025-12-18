#!/usr/bin/env python3
"""
run_gemm_hls_dse.py

- Sweeps pragma configs (unroll factor on K loop, pipeline II on Col loop).
- For each config, generates a small Tcl script for Vitis HLS.
- Runs synthesis.
- Parses the csynth.rpt.
- Writes results to CSV: gemm_hls_results.csv

Tested conceptually for Vitis HLS / Vivado HLS-style reports.
You may need to tweak regexes if Xilinx changes report formatting.
"""

import csv
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

# Name/path of the Vitis HLS binary.
VITIS_HLS_BIN = r"D:\Xilinx\2025.1\Vitis\bin\vitis-run.bat"   # change to full path if needed

# Your PYNQ-Z2 part.
FPGA_PART = "xc7z020clg400-1"

# Clock period (ns) — e.g., 10ns = 100 MHz
CLOCK_PERIOD = 10.0

# Kernel C++ file in this directory
KERNEL_CPP = "gemm.cpp"

# Output CSV
OUTPUT_CSV = "gemm_hls_results.csv"

# Design space: (config_name, unroll_k, pipeline_ii)
'''CONFIGS = [
    # name,    unroll_k, ii
    ("u1_ii1",  1,        1),
    ("u2_ii1",  2,        1),
    ("u4_ii1",  4,        1),
    ("u8_ii1",  8,        1),
    ("u2_ii2",  2,        2),
    ("u4_ii2",  4,        2),
    ("u8_ii2",  8,        2),

    # Add/remove configs as you like
]'''

CONFIGS = []

UNROLL_FACTORS = [1]
II_VALUES       = [1]

for u in UNROLL_FACTORS:
    for ii in II_VALUES:
        cfg_name = f"u{u}_ii{ii}"
        CONFIGS.append((cfg_name, u, ii))


# ---------------------------------------------------------------------
# Helpers to generate Tcl, run HLS, and parse reports
# ---------------------------------------------------------------------

def make_tcl_for_config(project_name: str, unroll_k: int, pipeline_ii: int) -> Path:
    """
    Create a Tcl script for one configuration.
    - project_name: name of the HLS project (also folder).
    - unroll_k: factor for K loop.
    - pipeline_ii: requested II for Col loop.

    Returns the Path to the generated Tcl script.
    """

    tcl_content = f"""\
# Auto-generated Tcl for {project_name}
open_project -reset {project_name}
set_top gemm

add_files {KERNEL_CPP}

open_solution "solution1"
set_part {{{FPGA_PART}}}
create_clock -period {CLOCK_PERIOD} -name default

# Apply directives:
# - pipeline the Col loop with II = {pipeline_ii}
# - unroll the K loop with factor = {unroll_k}
#
# NOTE: these paths rely on the loop labels in gemm.cpp:
#   Col: for (int j = 0; j < N; ++j) {{ ... }}
#   K:   for (int k = 0; k < N; ++k) {{ ... }}
# If you rename labels, update these paths.
set_directive_pipeline -II {pipeline_ii} gemm/Col
set_directive_unroll   -factor {unroll_k} gemm/K

csynth_design
exit
"""
    tcl_path = Path(f"run_{project_name}.tcl")
    tcl_path.write_text(tcl_content)
    return tcl_path


def run_vitis_hls(tcl_script: Path):
    """Invoke Vitis HLS (via vitis-run) with the given Tcl script."""
    print(f"[HLS] Running: {tcl_script}")
    try:
        subprocess.run(
            [VITIS_HLS_BIN, "--mode", "hls", "--tcl", str(tcl_script)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Vitis HLS failed for script {tcl_script}: {e}")
        raise



def _safe_int(text):
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_csynth_report(project_name: str, top_name: str = "gemm"):
    """
    Parse Vitis HLS csynth XML report for the given project.

    Looks for:
      <project_name>/solution1/syn/report/<top_name>_csynth.xml

    Returns a dict with:
      latency_min_cycles, latency_max_cycles,
      interval_min_ii, interval_max_ii,
      lut, ff, bram, dsp
    """

    proj_dir = Path(project_name)
    xml_path = proj_dir / "solution1" / "syn" / "report" / f"{top_name}_csynth.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"csynth XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # -------- Performance / Latency --------
    perf = root.find("PerformanceEstimates")
    latency_min_cycles = latency_max_cycles = None
    interval_min_ii = interval_max_ii = None

    if perf is not None:
        overall = perf.find("SummaryOfOverallLatency")
        if overall is not None:
            # These tag names are what Vitis HLS typically uses in csynth.xml.
            # If any come back None, it's just because your version uses a
            # slightly different name — you can open the XML and tweak them.
            latency_min_cycles = _safe_int(
                overall.findtext("Best-caseLatency")
            )
            latency_max_cycles = _safe_int(
                overall.findtext("Worst-caseLatency")
            )

            # These may or may not exist depending on version/design.
            interval_min_ii = _safe_int(
                overall.findtext("Best-caseInterval")
            )
            interval_max_ii = _safe_int(
                overall.findtext("Worst-caseInterval")
            )

    # -------- Resource utilization --------
    lut = ff = bram = dsp = None
    area = root.find("AreaEstimates")
    if area is not None:
        res = area.find("Resources")
        if res is not None:
            lut = _safe_int(res.findtext("LUT"))
            ff = _safe_int(res.findtext("FF"))
            # BRAM tag can be BRAM_18K or BRAM depending on version
            bram = _safe_int(
                res.findtext("BRAM_18K") or res.findtext("BRAM")
            )
            # DSP tag can be DSP or DSP48E depending on version
            dsp = _safe_int(
                res.findtext("DSP") or res.findtext("DSP48E")
            )

    return {
        "latency_min_cycles": latency_min_cycles,
        "latency_max_cycles": latency_max_cycles,
        "interval_min_ii": interval_min_ii,
        "interval_max_ii": interval_max_ii,
        "lut": lut,
        "ff": ff,
        "bram": bram,
        "dsp": dsp,
    }


# ---------------------------------------------------------------------
# Main DSE driver
# ---------------------------------------------------------------------

def main():
    root = Path(".").resolve()
    print(f"[INFO] Running GEMM HLS DSE in: {root}")

    rows = []

    for cfg_name, unroll_k, pipeline_ii in CONFIGS:
        project_name = f"proj_gemm_{cfg_name}"

        # 1) Generate Tcl
        tcl_script = make_tcl_for_config(
            project_name=project_name,
            unroll_k=unroll_k,
            pipeline_ii=pipeline_ii,
        )

        # 2) Run HLS
        try:
            run_vitis_hls(tcl_script)
        except Exception as e:
            print(f"[WARN] Skipping {cfg_name} due to HLS failure: {e}")
            continue

        # 3) Parse report
        try:
            metrics = parse_csynth_report(project_name)
        except Exception as e:
            print(f"[WARN] Failed to parse report for {cfg_name}: {e}")
            continue

        row = {
            "config_name": cfg_name,
            "unroll_k": unroll_k,
            "pipeline_ii": pipeline_ii,
            "latency_min_cycles": metrics["latency_min_cycles"],
            "latency_max_cycles": metrics["latency_max_cycles"],
            "interval_min_ii": metrics["interval_min_ii"],
            "interval_max_ii": metrics["interval_max_ii"],
            # keep the CSV column names as you like, but map from the lowercase keys:
            "LUT": metrics["lut"],
            "FF": metrics["ff"],
            "BRAM_18K": metrics["bram"],
            "DSP48E": metrics["dsp"],
        }

        rows.append(row)

        print(f"[OK] {cfg_name}: "
              f"latency_min={row['latency_min_cycles']}, "
              f"LUT={row['LUT']}, FF={row['FF']}, "
              f"BRAM={row['BRAM_18K']}, DSP={row['DSP48E']}")

    # 4) Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[RESULT] Wrote {len(rows)} rows to {OUTPUT_CSV}")
    else:
        print("[RESULT] No successful configurations to write.")


if __name__ == "__main__":
    main()
