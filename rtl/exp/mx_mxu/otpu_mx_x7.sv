// EXPERIMENTAL (docs/mx_mxu.md): not in any file list, not in the board build.
//
// The MX MXU column on Xilinx 7-series (xc7k480t, DSP48E1): one physical column whose 128 DSPs
// run at twice the core clock and serve two stationary rows (rtl/exp/dsp_pump's otpu_pump_pe and
// schedule), and two accumulating column ends (otpu_mx_colend) at the core clock. The UltraScale+
// version of the same modules is otpu_mx_us.sv (INT8 packing on DSP48E2 instead of pumping); a
// file list takes one of the two.
//
// Numerics (docs/mx_mxu.md, the ISA proposal): per stationary row r, weight row n, block k,
// sub-block b (32 elements):
//   c   = sum_{i in b} a_i * w_i                                (exact, today's chains)
//   sh  = Wf - dw[n,k] - da[r,k,b]   (Wf 11: W4X, 6-bit m_b; Wf 16: W8X, m_b = 1)
//   B   = sh >= 0 ? m_b << sh : 0                               (a flushed term: 0)
//   acc = sum_k sum_b B * c                                     (exact, 48 bits)
// The column end multiplies the sub-block's two chain ends (pre-added) by B and sums the four
// sub-blocks down the PCIN cascade as otpu_colend does; a fifth DSP accumulates the blocks of a
// row (P = P + PCIN; a row's first block: P = PCIN).

package otpu_mx_fam;
  // a column's latency: chunk t's row result (acc0 / acc1 / av) in cycle t + LATC + COL
  localparam int LATC = 16;
  localparam bit PUMPED = 1'b1;
endpackage

// The clocks: one MMCME2_ADV (VCO 1066.67 MHz from 133.33), clk_s = /8 (133.33), clk_f = /4
// (266.67), both through BUFGs (as dsp_pump_top).
module otpu_mx_clk (
  input  logic clk_in,
  output logic clk_s,
  output logic clk_f,
  output logic locked
);
  logic clkfb, clk_s_u, clk_f_u;
  MMCME2_ADV #(
    .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
    .CLKFBOUT_MULT_F(8.0), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(7.5),
    .CLKOUT0_DIVIDE_F(8.0), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
    .CLKOUT1_DIVIDE(4), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
  ) u_mmcm (
    .CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
    .CLKFBOUTB(), .CLKOUT0(clk_s_u), .CLKOUT0B(), .CLKOUT1(clk_f_u), .CLKOUT1B(),
    .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
    .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
  BUFG u_bs (.I(clk_s_u), .O(clk_s));
  BUFG u_bf (.I(clk_f_u), .O(clk_f));
endmodule

// The shared weight skew of the pumped array: position i delayed ceil(k/2), k = i % CLS.
module otpu_mx_front #(parameter int D = 128, parameter int CLS = 16) (
  input  logic               clk_s,
  input  logic [D-1:0][15:0] w,
  output logic [D-1:0][15:0] wc
);
  for (genvar i = 0; i < D; i++) begin : g_w
    otpu_skew #(.W(16), .N((i % CLS + 1) / 2)) u_w (.clk(clk_s), .en(1'b1), .d(w[i]), .q(wc[i]));
  end
endmodule

// The accumulating column end of one row (DSP48E1). DSP b (b = 0..3) registers the sub-block's
// chain ends (a[b] on A, d[b] on D) and B = bm[b] at edge E (the end of the cycle they are
// presented in), pre-adds the chain ends, multiplies by B (M at E+1), and the four products run
// down the cascade (P = PCIN + M; DSP 3 registers the block sum at E+2), as otpu_colend. DSP 4
// adds the block sum to its own P (X = P, Z = PCIN; OPMODE registered: `first`, presented in the
// cycle after the one that ends at E+1, makes it X = 0) at E+3. acc is DSP 4's P.
module otpu_mx_colend (
  input  logic              clk,
  input  logic [3:0][23:0]  a,
  input  logic [3:0][23:0]  d,
  input  logic [3:0][16:0]  bm,       // B = m_b << sh_b (unsigned), with a / d
  input  logic              first,    // two cycles after a / d: the block starts its row
  output logic [47:0]       acc
);
  logic [3:0][47:0] pc, pf;
  for (genvar b = 0; b < 4; b++) begin : g_d
    DSP48E1 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
      .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1), .CREG(0), .DREG(1),
      .ADREG(0), .MREG(1), .PREG(b == 3 ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),
      .A({{6{a[b][23]}}, a[b]}), .B({1'b0, bm[b]}), .C(48'd0), .D({d[b][23], d[b]}),
      .INMODE(5'b00100), .OPMODE(b == 0 ? 7'b000_01_01 : 7'b001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(1'b1), .CEA2(1'b1), .CEB1(1'b1), .CEB2(1'b1), .CEC(1'b0), .CED(1'b1), .CEAD(1'b0),
      .CEM(1'b1), .CEP(b == 3), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
      .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(30'd0), .BCIN(18'd0), .PCIN(b == 0 ? 48'd0 : pc[b == 0 ? 0 : b - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[b]), .P(pf[b]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  end
  DSP48E1 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("FALSE"), .USE_MULT("NONE"),
    .USE_SIMD("ONE48"), .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0), .CREG(0), .DREG(0),
    .ADREG(0), .MREG(0), .PREG(1), .INMODEREG(0), .OPMODEREG(1), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_acc (
    .CLK(clk),
    .A(30'd0), .B(18'd0), .C(48'd0), .D(25'd0),
    .INMODE(5'b00000), .OPMODE(first ? 7'b001_00_00 : 7'b001_00_10),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0), .CED(1'b0), .CEAD(1'b0),
    .CEM(1'b0), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b1), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pc[3]), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(), .P(acc), .CARRYCASCOUT(), .MULTSIGNOUT(),
    .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
endmodule

// One physical column, two stationary rows (virtual columns 2p, 2p+1 = rows 0, 1 here), COL
// hops from the front: the pumped PE chains of otpu_pump_col (clk_f = 2 x clk_s) and two
// accumulating column ends at clk_s. Inputs of chunk t in cycle t (the activations and the side
// information unskewed; wc_prev already skewed by the front and COL hops). acc0 / acc1 hold the
// two rows' accumulators in cycle t + LATC + COL when av (the row's last block was chunk t).
module otpu_mx_col2 #(parameter int D = 128, parameter int CLS = 16, parameter int COL = 0) (
  input  logic                  clk_s,
  input  logic                  clk_f,
  input  logic                  rst_s,
  input  logic [D-1:0][7:0]     x0,
  input  logic [D-1:0][7:0]     x1,
  input  logic [D-1:0][15:0]    wc_prev,
  output logic [D-1:0][15:0]    wc,
  input  logic                  hsel,        // per command: the high stream (PAIR)
  input  logic [31:0]           sw_lo,       // scale words: dw [7:0], m_b [8+6b +: 6]
  input  logic [31:0]           sw_hi,
  input  logic [31:0]           da0,         // rows' sub-block exponent offsets (4 x 8)
  input  logic [31:0]           da1,
  input  logic                  fmt,         // 1: W8X (Wf 16, m_b = 1); 0: W4X (Wf 11)
  input  logic                  v,
  input  logic                  first,
  input  logic                  last,
  output logic [47:0]           acc0,
  output logic [47:0]           acc1,
  output logic                  av
);
  localparam int NCH = D / CLS;
  initial if (NCH != 8 || CLS != 16) $fatal(1, "otpu_mx_col2: D 128, CLS 16 only");

  // ---- the fast phase (otpu_pump_col)
  logic rst_f;
  (* dont_touch = "true" *) logic p;
  always_ff @(posedge clk_f) begin
    rst_f <= rst_s;
    p <= rst_f ? 1'b1 : ~p;
  end
  (* dont_touch = "true" *) logic [NCH-1:0] pe_c, po_c;
  always_ff @(posedge clk_f) begin
    pe_c <= {NCH{~p}};
    po_c <= {NCH{p}};
  end
  logic [D-1:0][15:0] hop;                     // (the last column's: unused, trimmed)
  always_ff @(posedge clk_s) hop <= wc_prev;
  assign wc = hop;

  logic signed [23:0] pl [NCH];
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic [47:0] pc [CLS];
    for (genvar k = 0; k < CLS; k++) begin : g_k
      localparam int I = c * CLS + k;
      logic [7:0] a0, a1;
      otpu_skew #(.W(8), .N((k + 1) / 2 + COL)) u_x0 (.clk(clk_s), .en(1'b1), .d(x0[I]), .q(a0));
      otpu_skew #(.W(8), .N(k / 2 + 1 + COL))   u_x1 (.clk(clk_s), .en(1'b1), .d(x1[I]), .q(a1));
      wire [7:0] w = hsel ? wc_prev[I][15:8] : wc_prev[I][7:0];
      logic signed [23:0] pk;
      otpu_pump_pe #(.FIRST(k == 0), .ODD(k % 2 == 1)) u_ppe (
        .clk_f, .pe(pe_c[c]), .po(po_c[c]), .x0(a0), .x1(a1), .w,
        .pcin(k == 0 ? 48'd0 : pc[k == 0 ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
      if (k == CLS - 1) begin : g_end
        assign pl[c] = pk;
      end
    end
  end
  logic [NCH-1:0][23:0] h;
  always_ff @(posedge clk_f)
    for (int c = 0; c < NCH; c++) if (po_c[c]) h[c] <= pl[c];
  logic [3:0][23:0] a0e, d0e, a1e, d1e;
  always_comb
    for (int b = 0; b < 4; b++) begin
      a0e[b] = pl[2*b]; d0e[b] = pl[2*b+1];
      a1e[b] = h[2*b];  d1e[b] = h[2*b+1];
    end

  // ---- the shift: row 0's chain ends meet its column end in cycle t + 10 + COL, row 1's a
  // cycle later; the scale fields arrive there through skews (selected by hsel at the input)
  wire [31:0] sws = hsel ? sw_hi : sw_lo;
  logic [32:0] si0, si1;                                    // {fmt, dw, m} skewed
  logic [31:0] dq0, dq1;
  otpu_skew #(.W(33), .N(10 + COL)) u_s0 (.clk(clk_s), .en(1'b1), .d({fmt, sws}), .q(si0));
  otpu_skew #(.W(1 + 32), .N(1)) u_s1 (.clk(clk_s), .en(1'b1), .d(si0), .q(si1));
  otpu_skew #(.W(32), .N(10 + COL)) u_d0 (.clk(clk_s), .en(1'b1), .d(da0), .q(dq0));
  otpu_skew #(.W(32), .N(11 + COL)) u_d1 (.clk(clk_s), .en(1'b1), .d(da1), .q(dq1));
  function automatic logic [3:0][16:0] bms(input logic [32:0] si, input logic [31:0] da);
    for (int b = 0; b < 4; b++) begin
      logic signed [9:0] sh;
      logic [5:0] m;
      sh = (si[32] ? 10'sd16 : 10'sd11) - 10'(si[7:0]) - 10'(da[8*b +: 8]);
      m = si[32] ? 6'd1 : si[8 + 6*b +: 6];
      bms[b] = (sh < 0) ? 17'd0 : (17'(m) << sh[4:0]);
    end
  endfunction
  // the row's first block, at the accumulator's OPMODE register (two cycles after a / d); the
  // last, where its P holds the row (four cycles after)
  logic f0, f1, l0, l1, vl;
  otpu_skew #(.W(1), .N(12 + COL)) u_f0 (.clk(clk_s), .en(1'b1), .d(v && first), .q(f0));
  otpu_skew #(.W(1), .N(1)) u_f1 (.clk(clk_s), .en(1'b1), .d(f0), .q(f1));
  otpu_skew #(.W(1), .N(16 + COL)) u_l (.clk(clk_s), .en(1'b1), .d(v && last), .q(vl));
  logic [47:0] p0, p1, q0, q0d, q1;
  otpu_mx_colend u_ce0 (.clk(clk_s), .a(a0e), .d(d0e), .bm(bms(si0, dq0)), .first(f0), .acc(p0));
  otpu_mx_colend u_ce1 (.clk(clk_s), .a(a1e), .d(d1e), .bm(bms(si1, dq1)), .first(f1), .acc(p1));
  always_ff @(posedge clk_s) begin
    q0 <= p0;                    // row 0's P: its row in cycle t + 14 + COL
    q0d <= q0;
    q1 <= p1;                    // row 1's: in t + 15 + COL
  end
  assign acc0 = q0d;
  assign acc1 = q1;
  assign av = vl;
endmodule
