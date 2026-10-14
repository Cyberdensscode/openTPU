"""Abstract SRAM macros for NanGate45 (FreePDK45): LEF, Liberty and a Verilog model.

NanGate45 has no memory compiler. OpenROAD-flow-scripts' designs use bsg_fakeram's macros
(fakeram45_*: one read/write port, 1RW); a banked openTPU TMEM needs one read and one write port
per bank per cycle (1R1W, otpu_tmem IMPL 1), and the one copy (otpu_tmem IMPL 3 TDP) a dual-port
RAM, one port reading or writing and the other reading (1RW1R), which bsg_fakeram does not make.
This generator makes all three, in bsg_fakeram's form (the same pin names, layers and power
straps), so the flow treats them like the fakeram45 macros:

    1RW    clk, ce_in, we_in, addr_in[A], wd_in[W], w_mask_in[W], rd_out[W]
    1R1W   clk, re_in, r_addr_in[A], we_in, w_addr_in[A], wd_in[W], w_mask_in[W], rd_out[W]
    1RW1R  clk, ce_in, we_in, addr_in[A], wd_in[W], w_mask_in[W], rd_out[W] (port 0: a read or
           a write), ce1_in, addr1_in[A], rd1_out[W] (port 1: a read)

All controls are active high. A read (ce_in & !we_in, re_in, ce1_in) registers mem[addr] into
rd_out at the rising edge, which holds otherwise; a write stores wd_in's bits where w_mask_in is 1.
A 1R1W read of the word written at the same edge is undefined (the model returns X). A 1RW1R
macro reads first: port 0's write cycle registers the word before the write into rd_out, and a
port-1 read of the word being written returns the word before it, as the FPGA's true dual-port
block RAM in READ_FIRST (otpu_tmem's TDP replays depend on it: docs/asic.md).

Model (fitted to the 24 fakeram45_* macros of ORFS's nangate45 platform, which bsg_fakeram made
with CACTI; the 128x256 and 64x256 macros, whose heights were tuned by hand, are left out):

    area   = 1.486 * W^1.022 * N^0.840 um^2            (1RW; median error 15%, worst 29%)
    clk->q = -0.0382 + 0.0345 log2(N) + 0.0598 W/100 ns  (worst 21%)
    period = -0.0378 + 0.0296 log2(N) + 0.0662 W/100 ns  (min_period; worst 13%)
    clk internal power = 4.569 * W^0.989 * N^0.485       (the library's units; worst 27%)

A 1R1W macro is 1.5x the 1RW's area (an 8T bit cell, about 1.3x, and a second decoder) and 1.2x
its clock power, with the same read timing; a 1RW1R macro (the same two-port cell and decoders,
write drivers on one port) the same as the 1R1W. Setup and hold are 0.05 ns on every input, input
capacitance 5 fF (clk 25 fF), as in fakeram45. The read data's outputs drive like NanGate45's
BUF_X4 (a compiled SRAM's output buffer: its delay and transition over the load at a 17 ps input
transition, its 242 fF max_capacitance), with clk->q the model's at 7.6 fF of load: fakeram45's
scalar clk->q, a 500 fF limit and a 0.227 ns transition there let a macro drive a 2-3 mm wire
unbuffered (the resizer saw no violation; the card's FIFO to the MXU's decode was 3.4 ns of
wire, docs/asic.md). Leakage is 0, as in fakeram45 (the power estimate is dynamic). Geometry:
the pins on the left edge on metal3 at a 0.28 um pitch (rd_out, wd_in, w_mask_in, then the
addresses and controls, 2.24 um between groups; every pin on the 0.14 um metal3 track grid),
VDD / VSS straps on metal4 at a 4.48 um pitch, metal1-metal3 obstructed except the pin column,
metal4 except the straps; the height fits the pins and the size is snapped to the 0.19 x 1.4 um
site.

    python3 tools/asic/fakeram.py OUT_DIR 1r1w:8192x32 1rw:1024x64 1rw1r:4096x32 ...
                                                                    (N x W: depth x width)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

PITCH = 0.28          # pin pitch on metal3 (um)
PIN_W = 0.07          # pin size
GAP = 2.24            # between pin groups (16 metal3 tracks: every pin stays on the track grid)
MARGIN = 2.8          # top and bottom keep-out for the straps' ends
SITE_W, SITE_H = 0.19, 1.4
STRAP_W, STRAP_PITCH = 0.28, 4.48
SETUP = HOLD = 0.05
CAP_IN, CAP_CLK = 5.0, 25.0
# the read outputs' drive: NanGate45 BUF_X4's Z (NangateOpenCellLibrary_typical: cell_rise / fall,
# rise / fall_transition at an input transition of 0.0171859 ns, the tables' third row, over its
# load index) and its max_capacitance; clk->q is the delay at LOAD_REF
DRV_LOAD = (0.365616, 7.57219, 15.1444, 30.2888, 60.5775, 121.155, 242.31)
DRV_RISE = (0.0193796, 0.0243348, 0.0290476, 0.0382822, 0.0566239, 0.0931423, 0.166005)
DRV_FALL = (0.0258174, 0.0296289, 0.0328068, 0.0382686, 0.0480393, 0.0666218, 0.103403)
DRV_RTR = (0.00410698, 0.00765819, 0.0117212, 0.0202394, 0.0375246, 0.0721523, 0.141414)
DRV_FTR = (0.00383059, 0.00564352, 0.00744274, 0.0110284, 0.0184481, 0.0339994, 0.0658447)
MAX_CAP, LOAD_REF = 242.31, 1
R1W1_AREA, R1W1_POWER = 1.5, 1.2          # two-port (1R1W, 1RW1R) against 1RW
KINDS = ("1rw", "1r1w", "1rw1r")


def _snap(v: float, step: float) -> float:
    return math.ceil(round(v / step, 6)) * step


@dataclass(frozen=True)
class Macro:
    kind: str           # "1rw" | "1r1w" | "1rw1r"
    depth: int          # words
    width: int          # bits per word

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r}: one of {', '.join(KINDS)}")
        if self.depth < 2 or self.width < 1:
            raise ValueError(f"{self.depth}x{self.width}: depth >= 2, width >= 1")

    @property
    def name(self) -> str:
        return f"fakeram45_{self.kind}_{self.depth}x{self.width}"

    @property
    def abits(self) -> int:
        return max(1, (self.depth - 1).bit_length())

    # ---- the model
    @property
    def area_model(self) -> float:
        a = 1.486 * self.width ** 1.022 * self.depth ** 0.840
        return a * (1.0 if self.kind == "1rw" else R1W1_AREA)

    @property
    def clk_to_q(self) -> float:
        return round(max(0.10, -0.0382 + 0.0345 * math.log2(self.depth)
                         + 0.0598 * self.width / 100), 3)

    @property
    def min_period(self) -> float:
        return round(max(0.10, -0.0378 + 0.0296 * math.log2(self.depth)
                         + 0.0662 * self.width / 100), 3)

    @property
    def clk_power(self) -> float:
        p = 4.569 * self.width ** 0.989 * self.depth ** 0.485
        return round(p * (1.0 if self.kind == "1rw" else R1W1_POWER), 3)

    # ---- pins, in edge order: (group, [(name, direction)])
    def pin_groups(self) -> list[list[tuple[str, str]]]:
        w, a = self.width, self.abits
        bus = lambda n, k, d: [(f"{n}[{i}]", d) for i in range(k)]   # noqa: E731
        g = [bus("rd_out", w, "OUTPUT"), bus("wd_in", w, "INPUT"), bus("w_mask_in", w, "INPUT")]
        if self.kind == "1rw":
            g += [bus("addr_in", a, "INPUT"), [("we_in", "INPUT"), ("ce_in", "INPUT"),
                                                ("clk", "INPUT")]]
        elif self.kind == "1rw1r":
            g = [bus("rd_out", w, "OUTPUT"), bus("rd1_out", w, "OUTPUT")] + g[1:]
            g += [bus("addr_in", a, "INPUT"), bus("addr1_in", a, "INPUT"),
                  [("we_in", "INPUT"), ("ce_in", "INPUT"), ("ce1_in", "INPUT"), ("clk", "INPUT")]]
        else:
            g += [bus("r_addr_in", a, "INPUT"), bus("w_addr_in", a, "INPUT"),
                  [("re_in", "INPUT"), ("we_in", "INPUT"), ("clk", "INPUT")]]
        return g

    @property
    def size(self) -> tuple[float, float]:
        """(width, height) in um: the height fits the pins (at least a square's), the width
        gives the model's area, both snapped to the site."""
        groups = self.pin_groups()
        npins = sum(len(g) for g in groups)
        h_pins = npins * PITCH + (len(groups) - 1) * GAP + 2 * MARGIN
        h = _snap(max(h_pins, math.sqrt(self.area_model)), SITE_H)
        w = _snap(max(self.area_model / h, 2 * STRAP_PITCH + 2 * MARGIN), SITE_W)
        return round(w, 3), round(h, 3)

    def pin_rects(self) -> list[tuple[str, str, float]]:
        """(pin, direction, y of the pin's lower edge) from the bottom margin up."""
        out, y = [], MARGIN
        for gi, grp in enumerate(self.pin_groups()):
            if gi:
                y += GAP
            for name, d in grp:
                out.append((name, d, round(y, 3)))
                y += PITCH
        return out


def lef(m: Macro) -> str:
    w, h = m.size
    pins = m.pin_rects()
    L = ["VERSION 5.7 ;", 'BUSBITCHARS "[]" ;', f"MACRO {m.name}", f"  FOREIGN {m.name} 0 0 ;",
         "  SYMMETRY X Y R90 ;", f"  SIZE {w:.3f} BY {h:.3f} ;", "  CLASS BLOCK ;"]
    for name, d, y in pins:
        L += [f"  PIN {name}", f"    DIRECTION {d} ;", "    USE SIGNAL ;",
              "    SHAPE ABUTMENT ;", "    PORT", "      LAYER metal3 ;",
              f"      RECT 0.000 {y:.3f} {PIN_W:.3f} {y + PIN_W:.3f} ;", "    END",
              f"  END {name}"]
    straps = []
    x = MARGIN
    while x + STRAP_W <= w - MARGIN + 1e-9:
        straps.append(round(x, 3))
        x += STRAP_PITCH / 2
    for net, use, sel in (("VSS", "GROUND", 0), ("VDD", "POWER", 1)):
        L += [f"  PIN {net}", "    DIRECTION INOUT ;", f"    USE {use} ;", "    PORT",
              "      LAYER metal4 ;"]
        L += [f"      RECT {x0:.3f} {MARGIN:.3f} {x0 + STRAP_W:.3f} {h - MARGIN:.3f} ;"
              for i, x0 in enumerate(straps) if i % 2 == sel]
        L += ["    END", f"  END {net}"]
    L += ["  OBS", "    LAYER metal1 ;", f"    RECT 0 0 {w:.3f} {h:.3f} ;", "    LAYER metal2 ;",
          f"    RECT 0 0 {w:.3f} {h:.3f} ;", "    LAYER metal3 ;",
          f"    RECT {PIN_W:.3f} 0 {w:.3f} {h:.3f} ;"]
    y0 = 0.0
    for _, _, y in pins:                       # the pin column between the pins
        if y > y0:
            L.append(f"    RECT 0 {y0:.3f} {PIN_W:.3f} {y:.3f} ;")
        y0 = y + PIN_W
    L.append(f"    RECT 0 {y0:.3f} {PIN_W:.3f} {h:.3f} ;")
    L += ["    LAYER metal4 ;", f"    RECT 0 0 {w:.3f} {MARGIN:.3f} ;",
          f"    RECT 0 {h - MARGIN:.3f} {w:.3f} {h:.3f} ;"]
    x0 = 0.0
    for x in straps:
        if x > x0:
            L.append(f"    RECT {x0:.3f} {MARGIN:.3f} {x:.3f} {h - MARGIN:.3f} ;")
        x0 = x + STRAP_W
    L.append(f"    RECT {x0:.3f} {MARGIN:.3f} {w:.3f} {h - MARGIN:.3f} ;")
    L += ["  END", f"END {m.name}", "END LIBRARY", ""]
    return "\n".join(L)


def _constraints(indent: str) -> list[str]:
    out = []
    for kind, v in (("setup_rising", SETUP), ("hold_rising", HOLD)):
        out += [f"{indent}timing() {{", f"{indent}    related_pin : clk;",
                f"{indent}    timing_type : {kind} ;",
                f'{indent}    rise_constraint(scalar) {{ values ("{v:.3f}"); }}',
                f'{indent}    fall_constraint(scalar) {{ values ("{v:.3f}"); }}', f"{indent}}}"]
    return out


def _power(indent: str, v: float) -> list[str]:
    return [f"{indent}internal_power() {{", f'{indent}    rise_power(scalar) {{ values ("{v:.3f}"); }}',
            f'{indent}    fall_power(scalar) {{ values ("{v:.3f}"); }}', f"{indent}}}"]


def liberty(m: Macro) -> str:
    w, h = m.size
    n, pin_pow = m.name, round(m.clk_power / 100, 3)
    L = [f"library({n}) {{", "    technology (cmos);", "    delay_model : table_lookup;",
         '    time_unit : "1ns";', '    voltage_unit : "1V";', '    current_unit : "1uA";',
         '    leakage_power_unit : "1nw";', "    nom_process : 1;", "    nom_temperature : 25.000;",
         "    nom_voltage : 1.1;", "    capacitive_load_unit (1,ff);",
         '    pulling_resistance_unit : "1kohm";',
         "    operating_conditions(tt_1.0_25.0) {", "        process : 1;",
         "        temperature : 25.000;", "        voltage : 1.1;",
         "        tree_type : balanced_tree;", "    }",
         "    default_cell_leakage_power : 0;", "    default_fanout_load : 1;",
         "    default_inout_pin_cap : 0.0;", "    default_input_pin_cap : 0.0;",
         "    default_output_pin_cap : 0.0;", "    default_max_transition : 0.227;",
         "    default_operating_conditions : tt_1.0_25.0;",
         "    default_leakage_power_density : 0.0;", "    slew_derate_from_library : 1.000;"]
    for k in ("lower_threshold_pct_fall : 20.000", "upper_threshold_pct_fall : 80.000",
              "lower_threshold_pct_rise : 20.000", "upper_threshold_pct_rise : 80.000"):
        L.append(f"    slew_{k};")
    for k in ("input_threshold_pct_fall", "input_threshold_pct_rise", "output_threshold_pct_fall",
              "output_threshold_pct_rise"):
        L.append(f"    {k} : 50.000;")
    L += [f"    lu_table_template({n}_slew) {{", "        variable_1 : total_output_net_capacitance;",
          '        index_1 ("1000, 1001");', "    }"]
    for t, bits in (("DATA", m.width), ("ADDRESS", m.abits)):
        L += [f"    type ({n}_{t}) {{", "        base_type : array ;", "        data_type : bit ;",
              f"        bit_width : {bits};", f"        bit_from : {bits - 1};", "        bit_to : 0 ;",
              "        downto : true ;", "    }"]
    # the read data buses and their addresses, the write's address, the controls, the addresses
    if m.kind == "1rw":
        reads, waddr = [("rd_out", "addr_in")], "addr_in"
        ctrls, addrs = ["we_in", "ce_in"], ["addr_in"]
    elif m.kind == "1rw1r":
        reads, waddr = [("rd_out", "addr_in"), ("rd1_out", "addr1_in")], "addr_in"
        ctrls, addrs = ["we_in", "ce_in", "ce1_in"], ["addr_in", "addr1_in"]
    else:
        reads, waddr = [("rd_out", "r_addr_in")], "w_addr_in"
        ctrls, addrs = ["we_in", "re_in"], ["r_addr_in", "w_addr_in"]
    L += [f"    cell({n}) {{", f"        area : {w * h:.3f};", "        interface_timing : true;",
          "        memory() {", "            type : ram;", f"            address_width : {m.abits};",
          f"            word_width : {m.width};", "        }",
          "        pin(clk) {", "            direction : input;", f"            capacitance : {CAP_CLK:.3f};",
          "            clock : true;", f"            min_period : {m.min_period:.3f};"]
    L += _power("            ", m.clk_power)
    L += ["        }"]
    idx = ", ".join(f"{c:g}" for c in DRV_LOAD)
    tab = lambda name, vs: (f'                {name}({n}_slew) {{ index_1 ("{idx}"); '   # noqa: E731
                            f'values ("{", ".join(f"{v:.4f}" for v in vs)}"); }}')
    rise = [m.clk_to_q + d - DRV_RISE[LOAD_REF] for d in DRV_RISE]
    fall = [m.clk_to_q + d - DRV_FALL[LOAD_REF] for d in DRV_FALL]
    for rd, raddr in reads:
        L += [f"        bus({rd}) {{", f"            bus_type : {n}_DATA;",
              "            direction : output;", f"            max_capacitance : {MAX_CAP:.3f};",
              "            memory_read() {", f"                address : {raddr};", "            }",
              "            timing() {", '                related_pin : "clk";',
              "                timing_type : rising_edge;", "                timing_sense : non_unate;",
              tab("cell_rise", rise), tab("cell_fall", fall), tab("rise_transition", DRV_RTR),
              tab("fall_transition", DRV_FTR), "            }", "        }"]
    for c in ctrls:
        L += [f"        pin({c}) {{", "            direction : input;",
              f"            capacitance : {CAP_IN:.3f};"]
        L += _constraints("            ") + _power("            ", pin_pow) + ["        }"]
    for a in addrs:
        L += [f"        bus({a}) {{", f"            bus_type : {n}_ADDRESS;",
              "            direction : input;", f"            capacitance : {CAP_IN:.3f};"]
        L += _constraints("            ") + _power("            ", pin_pow) + ["        }"]
    for b in ("wd_in", "w_mask_in"):
        L += [f"        bus({b}) {{", f"            bus_type : {n}_DATA;",
              "            memory_write() {", f"                address : {waddr};",
              '                clocked_on : "clk";', "            }",
              "            direction : input;", f"            capacitance : {CAP_IN:.3f};"]
        L += _constraints("            ") + _power("            ", pin_pow) + ["        }"]
    L += ["        cell_leakage_power : 0.000;", "    }", "}", ""]
    return "\n".join(L)


def verilog(m: Macro, blackbox: bool = False) -> str:
    """The behavioural model (simulation), or with `blackbox` the port list only (synthesis:
    the cell comes from the Liberty)."""
    W, A = m.width, m.abits
    if m.kind == "1rw":
        ports = (f"  input clk,\n  input ce_in,\n  input we_in,\n  input [{A - 1}:0] addr_in,\n"
                 f"  input [{W - 1}:0] wd_in,\n  input [{W - 1}:0] w_mask_in,\n"
                 f"  output reg [{W - 1}:0] rd_out")
        body = (f"  reg [{W - 1}:0] mem [0:{m.depth - 1}];\n"
                "  always @(posedge clk)\n    if (ce_in) begin\n"
                "      if (we_in) mem[addr_in] <= (wd_in & w_mask_in) | (mem[addr_in] & ~w_mask_in);\n"
                "      else rd_out <= mem[addr_in];\n    end\n")
    elif m.kind == "1rw1r":
        # read first: port 0's write cycle, and a port-1 read of the word written, take the word
        # before the write
        ports = (f"  input clk,\n  input ce_in,\n  input we_in,\n  input [{A - 1}:0] addr_in,\n"
                 f"  input [{W - 1}:0] wd_in,\n  input [{W - 1}:0] w_mask_in,\n"
                 f"  output reg [{W - 1}:0] rd_out,\n  input ce1_in,\n"
                 f"  input [{A - 1}:0] addr1_in,\n  output reg [{W - 1}:0] rd1_out")
        body = (f"  reg [{W - 1}:0] mem [0:{m.depth - 1}];\n"
                "  always @(posedge clk) begin\n"
                "    if (ce_in) begin\n"
                "      rd_out <= mem[addr_in];\n"
                "      if (we_in) mem[addr_in] <= (wd_in & w_mask_in) | (mem[addr_in] & ~w_mask_in);\n"
                "    end\n"
                "    if (ce1_in) rd1_out <= mem[addr1_in];\n  end\n")
    else:
        ports = (f"  input clk,\n  input re_in,\n  input [{A - 1}:0] r_addr_in,\n  input we_in,\n"
                 f"  input [{A - 1}:0] w_addr_in,\n  input [{W - 1}:0] wd_in,\n"
                 f"  input [{W - 1}:0] w_mask_in,\n  output reg [{W - 1}:0] rd_out")
        body = (f"  reg [{W - 1}:0] mem [0:{m.depth - 1}];\n"
                "  always @(posedge clk) begin\n"
                "    if (we_in) mem[w_addr_in] <= (wd_in & w_mask_in) | (mem[w_addr_in] & ~w_mask_in);\n"
                "    if (re_in) rd_out <= (we_in && r_addr_in == w_addr_in) ? {"
                f"{W}{{1'bx}}}} : mem[r_addr_in];\n  end\n")
    attr = "(* blackbox *)\n" if blackbox else ""
    return f"{attr}module {m.name} (\n{ports}\n);\n{'' if blackbox else body}endmodule\n"


def write(m: Macro, out: Path) -> dict:
    """The macro's files in out/: <name>.lef, .lib, .v (model), _bb.v (black box)."""
    out.mkdir(parents=True, exist_ok=True)
    files = {"lef": out / f"{m.name}.lef", "lib": out / f"{m.name}.lib",
             "v": out / f"{m.name}.v", "bb": out / f"{m.name}_bb.v"}
    files["lef"].write_text(lef(m))
    files["lib"].write_text(liberty(m))
    files["v"].write_text(verilog(m))
    files["bb"].write_text(verilog(m, blackbox=True))
    return files


def parse(spec: str) -> Macro:
    """'1r1w:8192x32' -> Macro (depth x width)."""
    kind, size = spec.split(":")
    depth, width = size.lower().split("x")
    return Macro(kind, int(depth), int(width))


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        sys.exit(2)
    out = Path(argv[0])
    for spec in argv[1:]:
        m = parse(spec)
        write(m, out)
        w, h = m.size
        print(f"{m.name}: {w:.2f} x {h:.2f} um ({w * h:.0f} um^2), clk->q {m.clk_to_q} ns, "
              f"min period {m.min_period} ns")


if __name__ == "__main__":
    main(sys.argv[1:])
