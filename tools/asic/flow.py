"""openTPU on NanGate45 with OpenROAD (docs/asic.md): the ASIC block (rtl/asic/otpu_asic_top.sv)
through OpenROAD-flow-scripts, for PPA numbers.

    python3 tools/asic/flow.py prep POINT       # macros, the macro map, the ORFS design
    python3 tools/asic/flow.py config POINT     # the ORFS design (config.mk, SDC) only, again
    python3 tools/asic/flow.py run POINT        # ORFS: native (ORFS_HOME), else Docker
    python3 tools/asic/flow.py report POINT     # area, timing, power -> report.json
    python3 tools/asic/flow.py all POINT        # the three
    python3 tools/asic/flow.py points           # the design points

POINT is a design point: an ASIC target (targets/*.toml with kind "asic": its arch, uarch and
clock; card = nangate45, small = nangate45-small) or a variant of one (POINTS). Everything goes to
build/asic/<point>/:

    macros/        the SRAM macros the design uses (tools/asic/fakeram.py: LEF, Liberty, Verilog)
    src/           otpu_sram_map.sv (the leaves otpu_sram_1r1w / otpu_sram_1rw: one branch per
                   macro size), a variant's edited sources
    config.mk, constraint.sdc   the ORFS design
    work/          ORFS's WORK_HOME (logs/, reports/, results/, objects/)
    report.json    the PPA summary

run uses a native OpenROAD-flow-scripts where ORFS_HOME names one (a clone with its tools built:
asic/nangate45/README.md), else ORFS's Docker image (ORFS_IMAGE; on Apple silicon Docker Desktop
runs the amd64 image under Rosetta), which mounts the tree at its own path. prep elaborates the
design once with yosys and slang, with the leaves as parameterised black boxes, to list the RAM
tiles the design instantiates (kind, words, bits); the macros and the map follow from that list.
Its yosys: YOSYS, else ORFS_HOME's (slang built in), else yosys on PATH (with the slang plugin).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import fakeram  # noqa: E402
from opentpu import target as T  # noqa: E402

IMAGE = os.environ.get("ORFS_IMAGE", "openroad/orfs:latest")


def orfs_home() -> Path | None:
    """The native OpenROAD-flow-scripts ORFS_HOME names (checked: its env.sh and the openroad
    build_openroad.sh --local installs), or None: Docker."""
    h = os.environ.get("ORFS_HOME")
    if not h:
        return None
    p = Path(h).expanduser()
    if not (p / "env.sh").exists() or not (p / "tools/install/OpenROAD/bin/openroad").exists():
        raise RuntimeError(f"ORFS_HOME={h}: no built OpenROAD-flow-scripts there (env.sh, "
                           "tools/install/OpenROAD/bin/openroad; asic/nangate45/README.md)")
    return p


def yosys() -> str:
    """prep's yosys: YOSYS, else the native ORFS's own (slang built in), else yosys on PATH."""
    if os.environ.get("YOSYS"):
        return os.environ["YOSYS"]
    h = orfs_home()
    if h and (h / "tools/install/yosys/bin/yosys").exists():
        return str(h / "tools/install/yosys/bin/yosys")
    if not shutil.which("yosys"):
        raise RuntimeError("no yosys: set ORFS_HOME (ORFS's yosys) or YOSYS, or put yosys with "
                           "the slang plugin on PATH (asic/nangate45/README.md)")
    return "yosys"

# The design points: a target (its arch / uarch: otpu_asic_top's parameters; its core_mhz: the
# clock the SDC asks for), the floorplan's utilisation and global placement's density margin
# (addon: PLACE_DENSITY_LB_ADDON, 0.10 unless stated), the macros a script places before the
# macro placer (macros: asic/nangate45/<file>, ORFS's MACRO_PLACEMENT_TCL), and a variant's
# source edits (rewrite: file -> [(old, new)], applied to copies in src/). lm3la6: otpu_fpipe's
# multiplier and adder one and two stages deeper (LM 3, LA 6: the localparams of the VPU, the
# stream engine and the quantizer), for the critical-path study.
_DEEP = {f: [("localparam int LM = 2, LA = 4;", "localparam int LM = 3, LA = 6;")]
         for f in ("vpu/otpu_vpu.sv", "vpu/otpu_se_comp.sv", "vpu/otpu_se_tail.sv",
                   "vpu/otpu_quant.sv")}
POINTS = {
    "small": dict(target="nangate45-small", util=35),
    # the card: release r2's arch at ACT_ROWS 8 on a smaller die than the previous card point's
    # 35% (30.6 mm2, most of it empty), with the lower placement density that cleared that run's
    # congestion estimate and the MXU's macros by its columns (docs/asic.md: at 45% the macro
    # placer's corner block of ACT RAM left global placement at a congestion of 2.61, these 1.10),
    # the TMEM's by its readers 100 um right of them (tmem_gap, mxu_macros.tcl's TMEM_GAP_UM: the
    # macro placer's corner put the TMEM's reads 3-5 mm from their units and its clock late; 2.94 ns
    # -> 2.59 on the global route), and global routing's metal4 and up at 7% (fastroute)
    "card": dict(target="nangate45", util=45, addon=0.02, macros="mxu_macros.tcl", tmem_gap=100,
                 fastroute="fastroute.tcl"),
    "small-lm3la6": dict(target="nangate45-small", util=35, rewrite=_DEEP),
    "card-lm3la6": dict(target="nangate45", util=35, rewrite=_DEEP),
}
for _t in T.names():
    if T.load(_t).kind == "asic":
        POINTS.setdefault(_t, dict(target=_t, util=35))
ASIC_TOP = "asic/otpu_asic_top.sv"


def params(point: str) -> dict:
    """otpu_asic_top's parameters for the point's target: the sizes and uarch, and the optional
    ones where the target states them (MXU_E; TMEM_IMPL 3's knobs; SEQ_WC on; COLL_GATHER, the
    collective unit without GATHER as the card builds it). RPB 0 is every read lane
    (NRP x LANES, as opentpu.target.rtl_params)."""
    t = T.load(POINTS[point]["target"])
    a, u = t.arch, t.uarch
    p = dict(D=a["D"], MCOLS=a["MCOLS"], ACT_BLOCKS=a["ACT_BLOCKS"], ACT_ROWS=t.act_rows,
             TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
             FIFO_DEPTH=u["FIFO_DEPTH"], LANES=a["LANES"], WIN=u["WIN"],
             MXU_IMPL=u["MXU_IMPL"], MXU_CL=u["MXU_CL"], VPU_CL=u["VPU_CL"],
             ULANES=u["ULANES"], HAS_DSTEP=int(a["DSTEP"]), TMEM_IMPL=u["TMEM_IMPL"],
             RPB=u["RPB"] or T.NRP * a["LANES"], WPB=u["WPB"])
    if u.get("MXU_E"):
        p["MXU_E"] = u["MXU_E"]
    if u["TMEM_IMPL"] == 3:
        p.update({k: int(u[k]) for k in T.TMEM3_KEYS if k in u})
    if a["SEQ_WC"]:
        p["SEQ_WC"] = 1
    if "COLL_GATHER" in u:
        p["COLL_GATHER"] = int(u["COLL_GATHER"])
    return p


def clock_ns(point: str) -> float:
    return round(1000.0 / T.load(POINTS[point]["target"]).platform["core_mhz"], 4)


def out_dir(point: str) -> Path:
    return ROOT / "build" / "asic" / point


def sources(point: str) -> list[Path]:
    """The RTL the ASIC synthesizes: the target's (opentpu.target.sources: the core, the
    generic DSP cells, the macro-tiled RAMs), a variant's edited copies, the ASIC top."""
    src = out_dir(point) / "src"
    rw = POINTS[point].get("rewrite", {})
    files = [src / Path(f).name if f in rw else RTL / f
             for f in T.sources(POINTS[point]["target"])]
    return files + [RTL / ASIC_TOP]


def probe(point: str) -> list[fakeram.Macro]:
    """The RAM tiles the design instantiates: elaborate with the leaves as black boxes and read
    their parameters."""
    o = out_dir(point)
    stub = o / "src" / "probe_leaves.sv"
    stub.write_text(
        "(* blackbox *)\nmodule otpu_sram_1r1w #(parameter int W = 1, parameter int N = 2) (\n"
        "  input logic clk, re, we, input logic [$clog2(N)-1:0] ra, wa,\n"
        "  input logic [W-1:0] wd, wm, output logic [W-1:0] rd);\nendmodule\n"
        "(* blackbox *)\nmodule otpu_sram_1rw #(parameter int W = 1, parameter int N = 2) (\n"
        "  input logic clk, en, we, input logic [$clog2(N)-1:0] a,\n"
        "  input logic [W-1:0] wd, wm, output logic [W-1:0] rd);\nendmodule\n"
        "(* blackbox *)\nmodule otpu_sram_1rw1r #(parameter int W = 1, parameter int N = 2) (\n"
        "  input logic clk, ena, wea, enb, input logic [$clog2(N)-1:0] aa, ab,\n"
        "  input logic [W-1:0] wd, wm, output logic [W-1:0] rda, rdb);\nendmodule\n")
    gp = " ".join(f"-G {k}={v}" for k, v in params(point).items())
    files = " ".join(str(f) for f in sources(point) + [stub])
    dump = o / "src" / "probe_cells.txt"
    script = (f"read_slang -D SYNTHESIS --keep-hierarchy --ignore-assertions {files} "
              f"--top otpu_asic_top {gp}; tee -q -o {dump} dump t:otpu_sram_*")
    ys = yosys()
    r = subprocess.run([ys, "-m", "slang", "-q", "-p", script], capture_output=True, text=True)
    if r.returncode != 0 and "Can't load module" in r.stdout + r.stderr:   # slang built in (ORFS's)
        r = subprocess.run([ys, "-q", "-p", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yosys probe failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    tiles: dict[tuple, int] = {}
    for cell in re.split(r"\n\s*cell ", dump.read_text())[1:]:
        kind = re.match(r"\\otpu_sram_(\w+)", cell).group(1)
        w = int(re.search(r"parameter \\W (\d+)", cell).group(1))
        n = int(re.search(r"parameter \\N (\d+)", cell).group(1))
        tiles[(kind, n, w)] = tiles.get((kind, n, w), 0) + 1
    (o / "src" / "tiles.json").write_text(json.dumps(
        [{"kind": k, "words": n, "bits": w, "count": c} for (k, n, w), c in sorted(tiles.items())],
        indent=1))
    return [fakeram.Macro(k, n, w) for (k, n, w) in sorted(tiles)]


def sram_map(macros: list[fakeram.Macro]) -> str:
    """The leaves: one generate branch per macro size."""
    out = ["// Generated by tools/asic/flow.py: the leaves of rtl/tech/asic/otpu_tech_ram.sv on this",
           "// design's macros (tools/asic/fakeram.py).", ""]
    for kind in ("1r1w", "1rw", "1rw1r"):
        ms = [m for m in macros if m.kind == kind]
        if kind == "1rw1r" and not ms:      # (the third kind: only a design with one has it)
            continue
        if kind == "1r1w":
            out += ["module otpu_sram_1r1w #(parameter int W = 1, parameter int N = 2) (",
                    "  input  logic clk, re, we,",
                    "  input  logic [$clog2(N)-1:0] ra, wa,",
                    "  input  logic [W-1:0] wd, wm,", "  output logic [W-1:0] rd);"]
        elif kind == "1rw1r":
            out += ["module otpu_sram_1rw1r #(parameter int W = 1, parameter int N = 2) (",
                    "  input  logic clk, ena, wea, enb,",
                    "  input  logic [$clog2(N)-1:0] aa, ab,",
                    "  input  logic [W-1:0] wd, wm,", "  output logic [W-1:0] rda, rdb);"]
        else:
            out += ["module otpu_sram_1rw #(parameter int W = 1, parameter int N = 2) (",
                    "  input  logic clk, en, we,", "  input  logic [$clog2(N)-1:0] a,",
                    "  input  logic [W-1:0] wd, wm,", "  output logic [W-1:0] rd);"]
        for i, m in enumerate(ms):
            cond = f"{'end else ' if i else ''}if (W == {m.width} && N == {m.depth}) begin : g_{m.depth}x{m.width}"
            out.append(f"  {cond}")
            if kind == "1r1w":
                out.append(f"    {m.name} u_m (.clk, .re_in(re), .r_addr_in(ra), .we_in(we), "
                           ".w_addr_in(wa), .wd_in(wd), .w_mask_in(wm), .rd_out(rd));")
            elif kind == "1rw1r":
                out.append(f"    {m.name} u_m (.clk, .ce_in(ena), .we_in(wea), .addr_in(aa), "
                           ".wd_in(wd), .w_mask_in(wm), .rd_out(rda), .ce1_in(enb), "
                           ".addr1_in(ab), .rd1_out(rdb));")
            else:
                out.append(f"    {m.name} u_m (.clk, .ce_in(en), .we_in(we), .addr_in(a), "
                           ".wd_in(wd), .w_mask_in(wm), .rd_out(rd));")
        if ms:
            out += ["  end else begin : g_none",
                    f"    initial $fatal(1, \"otpu_sram_{kind}: no macro for %0d x %0d\", N, W);",
                    "  end"]
        else:
            out.append(f"  initial $fatal(1, \"otpu_sram_{kind}: no macros of this kind\");")
        out += ["endmodule", ""]
    return "\n".join(out)


def sdc(point: str) -> str:
    p = clock_ns(point)
    return f"""# openTPU ASIC block ({point}): one clock; the memory ports and controls timed against a
# virtual clock with 20% of the period outside (tools/asic/flow.py). No hold checks at the ports:
# the ideal virtual clock against the propagated core clock (~1.2 ns of insertion delay) makes
# the input paths hold violations (nangate45-small: 2188 endpoints, up to 0.69 ns: port
# A's read data into the MXU FIFOs' macros; 12 K hold buffers), which is the integrator's
# boundary, not the block's. After CTS vclk takes the core clock's insertion delay as its source
# latency (asic/nangate45/io_latency.tcl, ORFS POST_CTS_TCL), so the ports' slacks are
# latency-neutral; before it both clocks are ideal.
current_design otpu_asic_top
set clk_period {p}
create_clock -name core_clock -period $clk_period [get_ports clk]
create_clock -name vclk -period $clk_period
set_input_delay [expr $clk_period * 0.2] -clock vclk [all_inputs -no_clocks]
set_output_delay [expr $clk_period * 0.2] -clock vclk [all_outputs]
set_false_path -hold -from [all_inputs -no_clocks]
set_false_path -hold -to [all_outputs]
"""


def config_mk(point: str, macros: list[fakeram.Macro]) -> str:
    o = out_dir(point)
    P = POINTS[point]
    files = [str(f) for f in sources(point)] + [str(o / "src" / "otpu_sram_map.sv")] + \
            [str(o / "macros" / f"{m.name}_bb.v") for m in macros]
    lefs = " ".join(str(o / "macros" / f"{m.name}.lef") for m in macros)
    libs = " ".join(str(o / "macros" / f"{m.name}.lib") for m in macros)
    tops = " ".join(f"{k} {v}" for k, v in params(point).items())
    mpt = (f"export MACRO_PLACEMENT_TCL = {ROOT / 'asic' / 'nangate45' / P['macros']}\n"
           if P.get("macros") else "")
    if P.get("tmem_gap"):
        mpt += f"export TMEM_GAP_UM = {P['tmem_gap']}\n"
    if P.get("fastroute"):
        mpt += f"export FASTROUTE_TCL = {ROOT / 'asic' / 'nangate45' / P['fastroute']}\n"
    return f"""# Generated by tools/asic/flow.py: openTPU's ASIC block, design point {point}
export DESIGN_NAME = otpu_asic_top
export DESIGN_NICKNAME = otpu_{point}
export PLATFORM = nangate45
export SYNTH_HDL_FRONTEND = slang
export SYNTH_HIERARCHICAL = 1
export VERILOG_FILES = {' '.join(files)}
export VERILOG_TOP_PARAMS = {tops}
export SDC_FILE = {o / 'constraint.sdc'}
export ADDITIONAL_LEFS = {lefs}
export ADDITIONAL_LIBS = {libs}
export CORE_UTILIZATION = {P['util']}
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN = 5
export PLACE_DENSITY_LB_ADDON = {P.get('addon', 0.10)}
{mpt}export MACRO_PLACE_HALO = 10 10
# channels between macros at least the standard-cell grid's metal4 strap pitch (56 um), so the
# rails in every channel reach a strap (pdngen: "Unable to repair all channels" at 44.8)
export RTLMP_MIN_CHANNEL_SIZE = 60 60
# and no row narrower than that pitch (a pocket boxed in by macros: its rails would float)
export TAPCELL_TCL = {ROOT / 'asic' / 'nangate45' / 'tapcell.tcl'}
# after CTS the ports' virtual clock takes the core clock's insertion delay (latency-neutral ports)
export POST_CTS_TCL = {ROOT / 'asic' / 'nangate45' / 'io_latency.tcl'}
export TNS_END_PERCENT = 100
export LEC_CHECK = 0
# the small RAMs left to flip-flops (the dispatch window, the MXU's drain rows, LUT RAMs: the
# largest 8 Kbit at the card's point); the large ones are macros
export SYNTH_MEMORY_MAX_BITS = 16384
"""


def config(point: str) -> None:
    """config.mk and constraint.sdc again from prep's tiles (src/tiles.json), touching no
    source: a floorplan or placement setting changed, the synthesis kept."""
    o = out_dir(point)
    ts = json.loads((o / "src" / "tiles.json").read_text())
    macros = sorted({fakeram.Macro(x["kind"], x["words"], x["bits"]) for x in ts},
                    key=lambda m: (m.kind, m.depth, m.width))
    (o / "constraint.sdc").write_text(sdc(point))
    (o / "config.mk").write_text(config_mk(point, macros))


def prep(point: str) -> list[fakeram.Macro]:
    o = out_dir(point)
    (o / "src").mkdir(parents=True, exist_ok=True)
    for f, edits in POINTS[point].get("rewrite", {}).items():
        t = (RTL / f).read_text()
        for a, b in edits:
            if a not in t:
                raise RuntimeError(f"{point}: {f} has no {a!r}")
            t = t.replace(a, b)
        (o / "src" / Path(f).name).write_text(t)
    macros = probe(point)
    for m in macros:
        fakeram.write(m, o / "macros")
    (o / "src" / "otpu_sram_map.sv").write_text(sram_map(macros))
    (o / "constraint.sdc").write_text(sdc(point))
    (o / "config.mk").write_text(config_mk(point, macros))
    counts = json.loads((o / "src" / "tiles.json").read_text())
    n = sum(t["count"] for t in counts)
    bits = sum(t["count"] * t["words"] * t["bits"] for t in counts)
    print(f"{point}: {n} macros of {len(macros)} sizes, {bits / 8 / 1024:.0f} KiB of SRAM -> {o}")
    for t in counts:
        print(f"  {t['count']:3d} x {t['kind']} {t['words']} x {t['bits']}")
    return macros


def run(point: str, cpus: int | None = None, target: str = "finish") -> int:
    """ORFS's make on the point's design, up to target; cpus: ORFS's NUM_CORES (and the
    container's CPUs), by default every core natively, 6 in Docker."""
    orfs = orfs_home()
    if not orfs and not shutil.which("docker"):
        raise RuntimeError("no ORFS: set ORFS_HOME to a built OpenROAD-flow-scripts, or install "
                           "Docker for its image (asic/nangate45/README.md)")
    cpus = cpus or (os.cpu_count() if orfs else 6)
    o = out_dir(point)
    (o / "work").mkdir(parents=True, exist_ok=True)
    # ORFS_CLEAN=floorplan (or place, cts, ...): redo from that stage (config changes do not);
    # ORFS_VARS: more ORFS variables for this run, e.g. "SKIP_INCREMENTAL_REPAIR=1"
    stage = os.environ.get("ORFS_CLEAN")
    clean = (f"make DESIGN_CONFIG={o / 'config.mk'} WORK_HOME={o / 'work'} clean_{stage} && "
             if stage else "")
    make = (f"{clean}make DESIGN_CONFIG={o / 'config.mk'} WORK_HOME={o / 'work'} "
            f"NUM_CORES={cpus} {os.environ.get('ORFS_VARS', '')} {target}")
    if orfs:
        cmd = ["bash", "-c",
               f"cd {shlex.quote(str(orfs))} && source env.sh >/dev/null 2>&1; cd flow && {make}"]
    else:
        cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", f"--cpus={cpus}",
               "-v", f"{ROOT}:{ROOT}", "-w", str(ROOT), IMAGE, "bash", "-lc",
               f"cd /OpenROAD-flow-scripts && source env.sh >/dev/null 2>&1; cd flow && {make}"]
    with open(o / "run.log", "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def report(point: str) -> dict:
    """Area, timing and power from ORFS's final reports and metrics."""
    o = out_dir(point)
    w = o / "work"
    rep: dict = {"point": point, "target": POINTS[point]["target"],
                 "clock_ns": clock_ns(point), "params": params(point)}
    mets = sorted(w.glob("logs/nangate45/*/base/*.json")) + \
        sorted(w.glob("reports/nangate45/*/base/*.json"))
    for f in sorted(w.glob("logs/nangate45/*/base/*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for k, v in d.items():
            if any(s in k for s in ("design__instance__area", "design__instance__count",
                                    "timing__setup__ws", "timing__setup__tns",
                                    "power__total", "design__die__area", "design__core__area",
                                    "design__instance__utilization")):
                rep.setdefault("metrics", {})[f"{f.stem}:{k}"] = v
    # the latest stage's report: timing (worst slack with the ports' budgets; the clock's
    # minimum period, register to register), power (vectorless, the default activities), area
    rdir = w / "reports" / "nangate45" / f"otpu_{point}" / "base"
    for name in ("6_finish.rpt", "5_global_route.rpt", "4_cts_final.rpt"):
        if (rdir / name).exists():
            rep["stage"] = name.split(".")[0]
            t = (rdir / name).read_text()
            break
    else:
        t = ""
    m = re.search(r"report_worst_slack\s*-+\s*worst slack (?:max )?(-?[\d.]+)", t)
    if m:
        rep["wns_ns"] = float(m.group(1))
        rep["fmax_io_mhz"] = round(1000 / (rep["clock_ns"] - min(0.0, rep["wns_ns"])), 1)
    m = re.search(r"core_clock period_min = ([\d.]+) fmax = ([\d.]+)", t)
    if m:
        rep["period_min_ns"], rep["fmax_mhz"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", t, re.M)
    if m:
        rep["power_w"] = {"internal": float(m.group(1)), "switching": float(m.group(2)),
                          "leakage": float(m.group(3)), "total": float(m.group(4))}
        for g in ("Sequential", "Combinational", "Clock", "Macro"):
            m = re.search(rf"^{g}\s+\S+\s+\S+\s+\S+\s+([\d.eE+-]+)", t, re.M)
            if m:
                rep["power_w"][g.lower()] = float(m.group(1))
    m = re.search(r"Design area ([\d.]+) u\^2 (\d+)% utilization", t)
    if m:
        rep["design_area_um2"], rep["utilization_pct"] = float(m.group(1)), int(m.group(2))
    # area per unit (synthesis, before placement's buffers): the slice's units and the collective
    # unit, each with its macros (the otpu_sram_* leaves below it)
    st = rdir / "synth_stat.txt"
    if st.exists():
        units: dict = {}
        for sec in re.split(r"^=== ", st.read_text(), flags=re.M)[1:]:
            mod, _, path = sec.split(" ===", 1)[0].partition("$")
            p = path.split(".")[1:]
            if p[:1] == ["u_slice"]:
                p = p[1:]
            m = re.search(r"Chip area for module .*?: ([\d.]+)", sec)
            if len(p) != 1 or not m:
                continue
            # (yosys stat: count and area with the submodules, then without)
            mac = sum(float(x.group(1)) for x in re.finditer(
                r"^\s+\d+\s+([\d.E+-]+)\s+\S+\s+\S+\s+fakeram45_", sec, re.M))
            units[p[0]] = {"um2": round(float(m.group(1))), "macro_um2": round(mac)}
        rep["units"] = dict(sorted(units.items(), key=lambda kv: -kv[1]["um2"]))
    tiles = o / "src" / "tiles.json"
    if tiles.exists():
        ts = json.loads(tiles.read_text())
        rep["macros"] = sum(t["count"] for t in ts)
        rep["sram_kib"] = sum(t["count"] * t["words"] * t["bits"] for t in ts) / 8 / 1024
        rep["macro_area_um2"] = round(sum(
            t["count"] * fakeram.Macro(t["kind"], t["words"], t["bits"]).size[0]
            * fakeram.Macro(t["kind"], t["words"], t["bits"]).size[1] for t in ts))
    del mets
    (o / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("prep", "config", "run", "report", "all", "points"):
        print(__doc__)
        sys.exit(2)
    if argv[0] == "points":
        for k in POINTS:
            print(k, POINTS[k]["target"], f"{clock_ns(k)} ns", params(k))
        return
    point = argv[1]
    if point not in POINTS:
        sys.exit(f"unknown point {point}: {', '.join(POINTS)}")
    cpus = int(os.environ["ORFS_CPUS"]) if os.environ.get("ORFS_CPUS") else None
    if argv[0] in ("prep", "all"):
        prep(point)
    if argv[0] == "config":
        config(point)
    if argv[0] in ("run", "all"):
        rc = run(point, cpus, os.environ.get("ORFS_TARGET", "finish"))
        print(f"ORFS exit {rc}: {out_dir(point) / 'run.log'}")
        if rc:
            sys.exit(rc)
    if argv[0] in ("report", "all"):
        print(json.dumps(report(point), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
