// EXPERIMENTAL (docs/vpu_pump.md): a VPU lane's multiply-add unit pumped at twice the core clock.
//
// Today every VPU lane has its own slot-0 unit, otpu_fmma (y = a*b + c*e: two otpu_fmul, LM = 2
// stages, and an otpu_fadd, LA = 4 stages, at the core clock). A pumped unit takes two lanes'
// operands each core cycle and runs them one after the other through one unit at clk_f =
// 2 x clk_s: half the units for the same element rate, or twice the rate from the same units.
//
//   vp_pair_base   two otpu_fmma at clk_s (today)
//   vp_pair_pump   one unit at clk_f: a 2:1 operand mux into its input registers, the result
//                  into one of two holding registers by phase, both read by clk_s. DEEP = 0:
//                  today's otpu_fmma (6 stages); DEEP = 1: vp_fmma_deep with XR extra register
//                  stages after each of its six (for synthesis retiming to spread: 6 x (1 + XR)
//                  stages at clk_f); DEEP = 2: vp_fmma_split (the two long stages split by hand,
//                  10 stages at clk_f)
//
// Both advance on the VPU's pipeline enable `en` (a clk_s signal: a frozen core cycle freezes
// both of its fast cycles), so the pumped pair is the same per lane, cycle for cycle of `en`, as
// two units with a longer latency; tb_vpu_pump.sv checks the results bit for bit against the
// base pair (and both against each other with random enables).
//
// The phase: clk_s's toggle t_s sampled by clk_f (t_f); ph = t_s ^ t_f is 1 in the first fast
// cycle after a slow edge and 0 in the second (no reset needed: both clocks come from one MMCM
// at phase 0, every slow edge is a fast edge). The input registers take lane 0 at the mid edge
// (ph = 1) and lane 1 at the slow edge (ph = 0); with an even fast latency L, lane 0's result
// is in the unit's output register during a second half (ph = 0) and lane 1's during a first.
module vp_fmma_deep
  import otpu_fp::*;
#(parameter int XR = 1) (
  input  logic clk,
  input  logic en,
  input  f32_t a, b, c, e,
  output f32_t y
);
  // today's six register stages (fmul s1 | s2 in parallel for both products, fadd s1 .. s4),
  // each followed by XR plain registers that retiming may pull into the logic
  fmul_mid_t m_ab, m_ce;
  f32_t p_ab, p_ce;
  fadd_p1_t s1;
  fadd_p2_t s2;
  fadd_nm_t s3;
  f32_t r;
  fmul_mid_t dm_ab [XR + 1], dm_ce [XR + 1];
  f32_t dp_ab [XR + 1], dp_ce [XR + 1], dr [XR + 1];
  fadd_p1_t d1 [XR + 1];
  fadd_p2_t d2 [XR + 1];
  fadd_nm_t d3 [XR + 1];
  always_ff @(posedge clk) if (en) begin
    dm_ab[0] <= fp_mul_s1(a, b);
    dm_ce[0] <= fp_mul_s1(c, e);
    for (int k = 1; k <= XR; k++) begin
      dm_ab[k] <= dm_ab[k-1];
      dm_ce[k] <= dm_ce[k-1];
    end
    dp_ab[0] <= fp_mul_s2(dm_ab[XR]);
    dp_ce[0] <= fp_mul_s2(dm_ce[XR]);
    for (int k = 1; k <= XR; k++) begin
      dp_ab[k] <= dp_ab[k-1];
      dp_ce[k] <= dp_ce[k-1];
    end
    d1[0] <= fp_add_s1(dp_ab[XR], dp_ce[XR]);
    for (int k = 1; k <= XR; k++) d1[k] <= d1[k-1];
    d2[0] <= fp_add_s2(d1[XR]);
    for (int k = 1; k <= XR; k++) d2[k] <= d2[k-1];
    d3[0] <= fp_add_s3(d2[XR]);
    for (int k = 1; k <= XR; k++) d3[k] <= d3[k-1];
    dr[0] <= fp_add_s4(d3[XR]);
    for (int k = 1; k <= XR; k++) dr[k] <= dr[k-1];
  end
  assign y = dr[XR];
endmodule

// The two stages that do not fit a 3.75 ns period on the -2 Kintex-7 (vp_fmma_deep at 266.67 MHz:
// the multiplier's rounding fp_mul_s2 and the adder's normalize fp_add_s3), each split in two by
// hand (synthesis retiming did not move them): bit for bit the same functions.
typedef struct packed {
  logic               sp;
  logic [31:0]        sv;
  logic               s;
  logic [23:0]        mm;
  logic               inc;        // the round-up
  logic [7:0]         e8;         // e0's low byte
  logic               ovf, e254, neg, zero, all1;
} vp_mul_r_t;

function automatic vp_mul_r_t vp_mul_s2a(input otpu_fp::fmul_mid_t m);
  vp_mul_r_t r;
  logic g, st;
  logic signed [10:0] e0;
  r = '0;
  r.sp = m.sp; r.sv = m.sv; r.s = m.s;
  if (m.p[47]) begin
    r.mm = m.p[47:24]; g = m.p[23]; st = |m.p[22:0];
  end else begin
    r.mm = m.p[46:23]; g = m.p[22]; st = |m.p[21:0];
  end
  e0 = m.e + 11'(m.p[47]);
  r.inc = g && (st || r.mm[0]);
  r.e8 = e0[7:0];
  r.ovf = e0 >= 11'sd255;
  r.e254 = e0 == 11'sd254;
  r.neg = e0 < 11'sd0;
  r.zero = e0 == 11'sd0;
  r.all1 = &r.mm;
  return r;
endfunction

function automatic otpu_fp::f32_t vp_mul_s2b(input vp_mul_r_t r);
  logic [24:0] mr;
  logic c;
  if (r.sp) return r.sv;
  mr = {1'b0, r.mm} + {24'd0, r.inc};
  c = mr[24];
  if (r.ovf || (r.e254 && c)) return {r.s, 8'hFF, 23'd0};
  if (r.neg) return {r.s, 31'd0};
  if (r.zero) return r.all1 ? {r.s, 8'd1, 23'd0} : {r.s, 31'd0};
  return {r.s, r.e8 + {7'd0, c}, mr[22:0]};
endfunction

typedef struct packed {
  logic        sp;
  logic [2:0]  sv;
  logic        sa;
  logic        sub;
  logic [7:0]  e;
  logic [1:0]  n43;       // norm28's first two shift levels (16, 8)
  logic [27:0] y;
} vp_add_h_t;

function automatic vp_add_h_t vp_add_s3a(input otpu_fp::fadd_p2_t q);
  vp_add_h_t h;
  h.sp = q.sp; h.sv = q.sv; h.sa = q.sa; h.sub = q.sub; h.e = q.e;
  h.y = q.sum; h.n43 = '0;
  if (h.y[27:12] == 0) begin h.n43[1] = 1'b1; h.y = h.y << 16; end
  if (h.y[27:20] == 0) begin h.n43[0] = 1'b1; h.y = h.y << 8; end
  return h;
endfunction

function automatic otpu_fp::fadd_nm_t vp_add_s3b(input vp_add_h_t h);
  otpu_fp::fadd_nm_t n;
  logic [4:0] k;
  logic [27:0] y;
  y = h.y; k = {h.n43, 3'b000};
  if (y[27:24] == 0) begin k[2] = 1'b1; y = y << 4; end
  if (y[27:26] == 0) begin k[1] = 1'b1; y = y << 2; end
  if (y[27] == 0)    begin k[0] = 1'b1; y = y << 1; end
  n.sp = h.sp;
  n.sv = h.sv;
  n.s = h.sa;
  if (h.sub && !y[27] && !h.sp) begin
    n.sp = 1'b1;
    n.sv = otpu_fp::FADD_SV_ZERO;
  end
  n.mn = {y[27:2], y[1] | y[0]};
  n.e = 9'(h.e) - 9'(k);
  return n;
endfunction

// today's unit with those two stages split and a second product register (the DSP's M and P):
// 10 stages at clk_f (5 core cycles), against today's 6 at clk_s
module vp_fmma_split
  import otpu_fp::*;
(
  input  logic clk,
  input  logic en,
  input  f32_t a, b, c, e,
  output f32_t y
);
  fmul_mid_t m1_ab, m1_ce, m2_ab, m2_ce;
  vp_mul_r_t r_ab, r_ce;
  f32_t p_ab, p_ce, y1, y2;
  fadd_p1_t s1;
  fadd_p2_t s2;
  vp_add_h_t s3a;
  fadd_nm_t s3;
  always_ff @(posedge clk) if (en) begin
    m1_ab <= fp_mul_s1(a, b);
    m1_ce <= fp_mul_s1(c, e);
    m2_ab <= m1_ab;
    m2_ce <= m1_ce;
    r_ab <= vp_mul_s2a(m2_ab);
    r_ce <= vp_mul_s2a(m2_ce);
    p_ab <= vp_mul_s2b(r_ab);
    p_ce <= vp_mul_s2b(r_ce);
    s1 <= fp_add_s1(p_ab, p_ce);
    s2 <= fp_add_s2(s1);
    s3a <= vp_add_s3a(s2);
    s3 <= vp_add_s3b(s3a);
    y1 <= fp_add_s4(s3);
    y2 <= y1;
  end
  assign y = y2;
endmodule

module vp_pair_base
  import otpu_fp::*;
(
  input  logic clk_s,
  input  logic en,
  input  f32_t a0, b0, c0, e0, a1, b1, c1, e1,
  output f32_t y0, y1
);
  otpu_fmma #(.LM(2), .LA(4)) u_0 (.clk(clk_s), .en, .a(a0), .b(b0), .c(c0), .e(e0), .y(y0));
  otpu_fmma #(.LM(2), .LA(4)) u_1 (.clk(clk_s), .en, .a(a1), .b(b1), .c(c1), .e(e1), .y(y1));
endmodule

module vp_pair_pump
  import otpu_fp::*;
#(parameter int DEEP = 0, parameter int XR = 1) (
  input  logic clk_s,
  input  logic clk_f,
  input  logic en,                  // clk_s: both fast cycles of a frozen core cycle freeze
  input  f32_t a0, b0, c0, e0, a1, b1, c1, e1,   // clk_s registers (the lanes' operands)
  output f32_t y0, y1               // clk_s registers
);
  localparam int L = DEEP == 2 ? 10 : DEEP == 1 ? 6 * (1 + XR) : 6;   // the unit's fast latency
  initial if (L % 2) $fatal(1, "vp_pair_pump: the unit's fast latency must be even");
  logic t_s, t_f;
  always_ff @(posedge clk_s) t_s <= ~t_s;
  always_ff @(posedge clk_f) t_f <= t_s;
  wire ph = t_s ^ t_f;              // 1: the first fast cycle of a core cycle
  // the input registers of the unit: lane 0 at the mid edge, lane 1 at the slow edge. en is
  // a clk_s register: it holds over both fast edges of its core cycle (a slow -> fast path of one
  // fast period, as are the operands of lane 0)
  f32_t ra, rb, rc, re, y;
  always_ff @(posedge clk_f) if (en) begin
    ra <= ph ? a0 : a1;
    rb <= ph ? b0 : b1;
    rc <= ph ? c0 : c1;
    re <= ph ? e0 : e1;
  end
  if (DEEP == 2) begin : g_split
    vp_fmma_split u_u (.clk(clk_f), .en, .a(ra), .b(rb), .c(rc), .e(re), .y);
  end else if (DEEP == 1) begin : g_deep
    vp_fmma_deep #(.XR(XR)) u_u (.clk(clk_f), .en, .a(ra), .b(rb), .c(rc), .e(re), .y);
  end else begin : g_flat
    otpu_fmma #(.LM(2), .LA(4)) u_u (.clk(clk_f), .en, .a(ra), .b(rb), .c(rc), .e(re), .y);
  end
  // lane 0's result in a second half (ph = 0), lane 1's in a first half
  f32_t h0, h1;
  always_ff @(posedge clk_f) if (en) begin
    if (!ph) h0 <= y;
    else     h1 <= y;
  end
  assign y0 = h0;
  assign y1 = h1;
endmodule

// the out-of-context top: the clocks from one MMCME2_ADV (VCO 133.33 x MULT, clk_s = VCO / DIV_S,
// clk_f = VCO / DIV_F) through BUFGs; NP pairs side by side; every input and output registered
// at clk_s. VAR 0: vp_pair_base; 1: vp_pair_pump (today's unit at clk_f); 2: vp_pair_pump with
// the deep unit (XR); 3: vp_pair_pump with the split unit.
module vpu_pump_top
  import otpu_fp::*;
#(
  parameter int VAR = 1,
  parameter int NP = 4,             // pairs (8 lanes: one VPU's slot 0)
  parameter int XR = 1,
  parameter real MULT = 8.0,
  parameter real DIV_S = 8.0,
  parameter int DIV_F = 4
) (
  input  logic clk_in,
  input  logic en_in,
  input  f32_t a_in [2*NP], b_in [2*NP], c_in [2*NP], e_in [2*NP],
  output f32_t y_out [2*NP]
);
  logic clkfb, clk_s_u, clk_f_u, clk_s, clk_f, locked;
`ifdef SYNTHESIS
  MMCME2_ADV #(
    .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
    .CLKFBOUT_MULT_F(MULT), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(7.5),
    .CLKOUT0_DIVIDE_F(DIV_S), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
    .CLKOUT1_DIVIDE(DIV_F), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
  ) u_mmcm (
    .CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
    .CLKFBOUTB(), .CLKOUT0(clk_s_u), .CLKOUT0B(), .CLKOUT1(clk_f_u), .CLKOUT1B(),
    .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
    .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
  BUFG u_bs (.I(clk_s_u), .O(clk_s));
  BUFG u_bf (.I(clk_f_u), .O(clk_f));
`else
  assign clk_s = clk_in;            // (simulation: tb_vpu_pump drives the pairs directly)
  assign clk_f = clk_in;
`endif
  (* max_fanout = 64 *) logic en;
  f32_t a [2*NP], b [2*NP], c [2*NP], e [2*NP], y [2*NP];
  always_ff @(posedge clk_s) begin
    en <= en_in;
    a <= a_in; b <= b_in; c <= c_in; e <= e_in;
    y_out <= y;
  end
  for (genvar p = 0; p < NP; p++) begin : g_p
    if (VAR == 0) begin : g_b
      vp_pair_base u_p (.clk_s, .en, .a0(a[2*p]), .b0(b[2*p]), .c0(c[2*p]), .e0(e[2*p]),
                        .a1(a[2*p+1]), .b1(b[2*p+1]), .c1(c[2*p+1]), .e1(e[2*p+1]),
                        .y0(y[2*p]), .y1(y[2*p+1]));
    end else begin : g_q
      vp_pair_pump #(.DEEP(VAR == 3 ? 2 : VAR == 2 ? 1 : 0), .XR(XR)) u_p (
        .clk_s, .clk_f, .en, .a0(a[2*p]), .b0(b[2*p]), .c0(c[2*p]), .e0(e[2*p]),
        .a1(a[2*p+1]), .b1(b[2*p+1]), .c1(c[2*p+1]), .e1(e[2*p+1]),
        .y0(y[2*p]), .y1(y[2*p+1]));
    end
  end
endmodule
