// EXPERIMENTAL (docs/mx_mxu.md): the out-of-context baseline for mx_top: today's MXU datapath
// (otpu_mxu IMPL 2) from the chunk and the ACT rows to the read-modify-write lanes, in the same
// surroundings as mx_top (inputs and outputs registered, no issuer, no chunk / scale FIFOs, no
// TMEM addresses, no RMAX). Copied from rtl/mxu/otpu_mxu.sv: the decode (wd), the weight skew, the
// systolic columns (otpu_pe / otpu_colend from rtl/tech/<family>, the column ends' u_mb skews,
// u_dsk), the per-column epilogue (i2f, x ws, x ascale, PAIR's add, the isum_4 partial loop, the
// combine), the result FIFO, the drain (NL = MCOLS lanes, a whole row a cycle) and its
// read-modify-write lanes (otpu_fmadd, u_rw). The clocks: otpu_mx_clk (clk_s only).
module base_top
  import otpu_fp::*;
#(
  parameter int MCOLS = 4,
  parameter int D = 128,
  parameter int CL = 16
) (
  input  logic                          clk_in,
  input  logic                          rst,
  input  logic [MCOLS-1:0][D-1:0][7:0]  x_in,
  input  logic [D-1:0][7:0]             chunk_in,
  input  logic [1:0]                    wf_in,
  input  logic                          h_in,
  input  logic [15:0]                   mb_lo_in,
  input  logic [15:0]                   mb_hi_in,
  input  logic [31:0]                   ws_lo_in,
  input  logic [31:0]                   ws_hi_in,
  input  logic [MCOLS-1:0][31:0]        as_in,
  input  logic [MCOLS-1:0]              hi_in,
  input  logic                          pair_in,
  input  logic [7:0]                    m_in,
  input  logic                          v_in,
  input  logic                          last_in,
  input  logic                          o_in,
  input  logic [MCOLS-1:0][31:0]        old_in,
  input  logic [MCOLS-1:0][31:0]        alpha_in,
  output logic [MCOLS-1:0][31:0]        y,
  output logic [MCOLS-1:0]              yv,
  output logic [MCOLS-1:0][23:0]        ytag,
  output logic                          busy
);
  localparam int NL = MCOLS;
  localparam int LM = 2, LA = 4;
  localparam int NPART = 4;
  localparam int RF = 64;
  localparam int RFW = $clog2(RF);
  localparam int MW = $clog2(MCOLS) + 1;
  localparam int CLS = (D / 4 < CL) ? D / 4 : CL;
  localparam int CPS = D / 4 / CLS;
  localparam int LDOT = CLS + MCOLS + 3;
  localparam int SW = 16 + $clog2(D);
  localparam int MG = SW - 1;
  localparam int LZW = $clog2(MG);

  logic clk, clk_f, locked;
  otpu_mx_clk u_clk (.clk_in, .clk_s(clk), .clk_f, .locked);
  logic [1:0] rs;
  always_ff @(posedge clk) rs <= {rs[0], rst | ~locked};
  wire rst_s = rs[1];
  wire en_c = 1'b1;

  // ---- R = S0: the inputs, registered (the chunk FIFO's read register, the ACT RAM's read,
  // the scale FIFO's read, the command's fields)
  typedef struct packed {
    logic       v;
    logic       last;
    logic       first;
    logic [1:0] q;
    logic       h;
    logic       o;
  } cm_t;
  logic [MCOLS*D*8-1:0] a0;
  logic [D*8-1:0] w0;
  logic [1:0] wf0;
  logic [15:0] mb0, mb0h;
  f32_t ws0, ws0h;
  f32_t as0 [MCOLS];
  logic [MCOLS-1:0] hi0;
  logic pr0;
  logic [7:0] M0r, mr;
  logic vr, lr, hr, or_;
  logic [NL-1:0][31:0] oldr, alr;
  always_ff @(posedge clk) begin
    a0 <= x_in; w0 <= chunk_in; wf0 <= wf_in; mb0 <= mb_lo_in; mb0h <= mb_hi_in;
    ws0 <= ws_lo_in; ws0h <= ws_hi_in; hi0 <= hi_in; pr0 <= pair_in; M0r <= m_in; mr <= m_in;
    vr <= v_in; lr <= last_in; hr <= h_in; or_ <= o_in; oldr <= old_in; alr <= alpha_in;
    for (int j = 0; j < MCOLS; j++) as0[j] <= as_in[j];
  end
  // the block index within its row (otpu_mxu's ck), for the partial loop's first / q
  logic [15:0] ck;
  always_ff @(posedge clk)
    if (rst_s) ck <= '0;
    else if (vr) ck <= lr ? '0 : ck + 16'd1;
  cm_t m0;
  always_comb begin
    m0 = '0;
    m0.v = vr;
    m0.last = lr;
    m0.first = (ck < 16'(NPART));
    m0.q = ck[1:0];
    m0.h = hr;
    m0.o = or_;
  end

  function automatic logic [7:0] dec4(input logic [3:0] c, input logic fp);
    if (!fp) return {{4{c[3]}}, c};
    case (c)
      4'd5: return 8'd6;
      4'd6: return 8'd8;
      4'd7: return 8'd12;
      4'd8: return 8'd0;
      4'd9: return -8'sd1;
      4'd10: return -8'sd2;
      4'd11: return -8'sd3;
      4'd12: return -8'sd4;
      4'd13: return -8'sd6;
      4'd14: return -8'sd8;
      4'd15: return -8'sd12;
      default: return 8'(c);
    endcase
  endfunction
  function automatic logic [7:0] wsel(input logic [D*8-1:0] w, input int i, input logic hb,
                                      input logic [1:0] wf);
    if (wf == 2'd0) return w[i*8 +: 8];
    return dec4(w[(hb ? D*4 : 0) + 4*i +: 4], wf == 2'd2);
  endfunction

  // ---- the systolic array (otpu_mxu g_sys)
  logic signed [SW-1:0] s4 [MCOLS];
  logic [D-1:0][15:0] wd;
  always_ff @(posedge clk) if (en_c)
    for (int i = 0; i < D; i++) wd[i] <= {wsel(w0, i, 1'b1, wf0), wsel(w0, i, m0.h, wf0)};
  logic [D-1:0][15:0] wc [MCOLS];
  for (genvar i = 0; i < D; i++) begin : g_wsk
    otpu_skew #(.W(16), .N(i % CLS)) u_w (.clk, .en(en_c), .d(wd[i]), .q(wc[0][i]));
  end
  always_ff @(posedge clk) if (en_c)
    for (int j = 1; j < MCOLS; j++) wc[j] <= wc[j-1];
  logic [7:0] as_k [MCOLS][D];
  logic signed [23:0] preg [MCOLS][D];
  logic [47:0] pc [MCOLS][D];
  for (genvar j = 0; j < MCOLS; j++) begin : g_ask
    for (genvar i = 0; i < D; i++) begin : g_p
      otpu_skew #(.W(8), .N(i % CLS + j)) u_a (.clk, .en(en_c), .d(a0[(j*D + i)*8 +: 8]),
                                             .q(as_k[j][i]));
      localparam int JW = (j == 0) ? 0 : j - 1;
      otpu_pe #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0)) u_pe (
        .clk, .en(en_c), .act(as_k[j][i]), .wlo(wc[JW][i][7:0]), .whi(wc[JW][i][15:8]),
        .sel(hi0[j]),
        .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
        .p(preg[j][i]));
    end
  end
  localparam int CPA = (CPS + 1) / 2;
  logic signed [SW-1:0] s4r [MCOLS];
  for (genvar j = 0; j < MCOLS; j++) begin : g_col_end
    logic [15:0] mbd;
    logic [3:0][23:0] ca, cd;
    otpu_skew #(.W(16), .N(CLS + 1 + j)) u_mb (.clk, .en(en_c), .d(hi0[j] ? mb0h : mb0),
                                             .q(mbd));
    always_comb
      for (int b = 0; b < 4; b++) begin
        ca[b] = '0; cd[b] = '0;
        for (int c = 0; c < CPS; c++)
          if (c < CPA) ca[b] = ca[b] + preg[j][(b*CPS + c)*CLS + CLS - 1];
          else cd[b] = cd[b] + preg[j][(b*CPS + c)*CLS + CLS - 1];
      end
    otpu_colend #(.SW(SW)) u_ce (.clk, .en(en_c), .a(ca), .d(cd), .mb(mbd), .s(s4r[j]));
    otpu_skew #(.W(SW), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(en_c), .d(s4r[j]), .q(s4[j]));
  end
  cm_t m4, m5, m6;
  f32_t ws4, ws4h, ws5, ws5h;
  otpu_delay #(.W($bits(cm_t)), .N(LDOT)) u_m4 (.clk, .en(en_c), .d(m0), .q(m4));
  otpu_delay #(.W(32), .N(LDOT)) u_ws4 (.clk, .en(en_c), .d(ws0), .q(ws4));
  otpu_delay #(.W(32), .N(LDOT)) u_ws4h (.clk, .en(en_c), .d(ws0h), .q(ws4h));

  // ---- S5, S6: int -> fp32 (otpu_mxu g_d2f)
  typedef struct packed {
    logic           z, s;
    logic [LZW-1:0] lz;
    logic [MG-1:0]  mag;
  } dmid_t;
  function automatic dmid_t d2f_s1(input logic signed [SW-1:0] x);
    dmid_t m;
    m.z   = (x == 0);
    m.s   = x[SW-1];
    m.mag = MG'(x[SW-1] ? -x : x);
    m.lz  = LZW'(lzc32(32'(m.mag) << (32 - MG)));
    return m;
  endfunction
  function automatic f32_t d2f_s2(input dmid_t m);
    logic [MG-1:0] nrm;
    if (m.z) return F_ZERO;
    nrm = m.mag << m.lz;
    return {m.s, 8'(126 + MG - int'(m.lz)), 23'(nrm[MG-2:0]) << (24 - MG)};
  endfunction
  f32_t fi [MCOLS];
  f32_t ws6 [MCOLS];
  dmid_t im [MCOLS];
  always_ff @(posedge clk) if (en_c) begin
    for (int j = 0; j < MCOLS; j++) im[j] <= d2f_s1(s4[j]);
    for (int j = 0; j < MCOLS; j++) fi[j] <= d2f_s2(im[j]);
    m5 <= m4; ws5 <= ws4; ws5h <= ws4h;
    m6 <= m5;
  end
  for (genvar j = 0; j < MCOLS; j++) begin : g_ws6
    always_ff @(posedge clk)
      if (rst_s) ws6[j] <= '0; else if (en_c) ws6[j] <= hi0[j] ? ws5h : ws5;
  end

  // ---- the epilogue (otpu_mxu, after the dot product)
  f32_t t1 [MCOLS], t2 [MCOLS], tq [MCOLS], pacc [MCOLS];
  cm_t  mt_p, mt, mq_p, mq, mx, ma;
  logic [7:0] M0;
  always_ff @(posedge clk) if (en_c) M0 <= M0r;
  otpu_delay #(.W($bits(cm_t)), .N(2 * LM - 1)) u_mt (.clk, .en(en_c), .d(m6), .q(mt_p));
  always_ff @(posedge clk) if (rst_s) mt <= '0; else if (en_c) mt <= mt_p;
  otpu_delay #(.W($bits(cm_t)), .N(LA - 1)) u_mq (.clk, .en(en_c), .d(mt), .q(mq_p));
  always_ff @(posedge clk) if (rst_s) mq <= '0; else if (en_c) mq <= mq_p;
  assign mx = pr0 ? mq : mt;
  otpu_delay #(.W($bits(cm_t)), .N(LA)) u_ma (.clk, .en(en_c), .d(mx), .q(ma));
  for (genvar j = 0; j < MCOLS; j++) begin : g_col
    f32_t fb, prev, as_p, asq, tx;
    otpu_delay #(.W(32), .N(LDOT + 2 + LM - 1)) u_as (.clk, .en(en_c), .d(as0[j]), .q(as_p));
    always_ff @(posedge clk) if (rst_s) asq <= '0; else if (en_c) asq <= as_p;
    otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en(en_c), .a(fi[j]), .b(ws6[j]), .y(t1[j]));
    otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en(en_c), .a(t1[j]), .b(asq), .y(t2[j]));
    if (j < MCOLS / 2) begin : g_pair
      f32_t pb;
      always_comb begin
        pb = F_ZERO;
        for (int m = 1; m <= MCOLS / 2; m++)
          if (j + m < MCOLS && mt.o && M0 == 8'(m)) pb = t2[j + m];
      end
      otpu_fadd #(.LAT(LA)) u_pr (.clk, .en(en_c), .a(t2[j]), .b(pb), .y(tq[j]));
      assign tx = pr0 ? tq[j] : t2[j];
    end else begin : g_nopair
      assign tq[j] = '0;
      assign tx = t2[j];
    end
    otpu_delay #(.W(32), .N(NPART - LA)) u_fb (.clk, .en(en_c), .d(pacc[j]), .q(fb));
    assign prev = mx.first ? F_ZERO : fb;
    otpu_fadd #(.LAT(LA)) u_acc (.clk, .en(en_c), .a(prev), .b(tx), .y(pacc[j]));
  end
  wire zb = !ma.v || ma.last || (ma.first && ma.q == 2'd0);
  logic mid_y;
  otpu_delay #(.W(1), .N(LA)) u_my (.clk, .en(en_c), .d(ma.v && !ma.last), .q(mid_y));
  f32_t pp1 [MCOLS], pb [MCOLS], cy [MCOLS], cy1 [MCOLS], rowv [MCOLS];
  always_ff @(posedge clk) if (en_c)
    for (int j = 0; j < MCOLS; j++) begin
      pp1[j] <= pacc[j];
      pb[j]  <= zb ? F_ZERO : pp1[j];
      cy1[j] <= mid_y ? cy[j] : F_ZERO;
    end
  wire launch = ma.v && ma.last;
  logic lv2;
  for (genvar j = 0; j < MCOLS; j++) begin : g_comb
    otpu_fadd #(.LAT(LA)) u_p (.clk, .en(en_c), .a(pacc[j]), .b(pb[j]), .y(cy[j]));
    otpu_fadd #(.LAT(LA)) u_c (.clk, .en(en_c), .a(cy1[j]), .b(cy[j]), .y(rowv[j]));
  end
  f32_t rowr [MCOLS];
  always_ff @(posedge clk) if (en_c) for (int j = 0; j < MCOLS; j++) rowr[j] <= rowv[j];
  otpu_delay #(.W(1), .N(2 * LA + 1)) u_lv (.clk, .en(en_c), .d(launch), .q(lv2));

  // ---- result FIFO (otpu_mxu's), with the row's index (as mx_top's)
  localparam int FW = MCOLS * 32 + 16;
  logic [RFW-1:0] rf_q, rf_t;
  logic [RFW:0]   rf_n;
  logic           rf_hs;
  logic [15:0]    ncnt;
  wire rf_push = en_c && lv2;
  logic [FW-1:0] rowp, rh0, rh1, rfh;
  logic [2:0][FW-1:0] rq;
  logic [2:0][RFW-1:0] rf_qc;
  always_comb begin
    for (int j = 0; j < MCOLS; j++) rowp[32*j +: 32] = rowr[j];
    rowp[MCOLS*32 +: 16] = ncnt;
    for (int c = 0; c < 3; c++) rf_qc[c] = rf_q + RFW'(c);
  end
  for (genvar c = 0; c < 3; c++) begin : g_rfm
    (* ram_style = "distributed" *) logic [FW-1:0] m [RF];
    always_ff @(posedge clk) if (rf_push) m[rf_t] <= rowp;
    assign rq[c] = (rf_push && rf_t == rf_qc[c]) ? rowp : m[rf_qc[c]];
  end
  always_ff @(posedge clk) begin
    rh0 <= rf_hs ? rq[1] : rq[0];
    rh1 <= rf_hs ? rq[2] : rq[1];
  end
  assign rfh = rf_hs ? rh1 : rh0;

  // ---- drain: the head row's M results a cycle into the RMW lanes (otpu_mxu's r0 / r1 /
  // fmadd / u_rw with old and alpha from the inputs)
  wire drain_go = (rf_n != 0);
  wire d_pop = drain_go;
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][23:0]    tg;
    logic [NL-1:0][31:0]    nv;
  } rmw_t;
  rmw_t r0, r1, rw;
  always_ff @(posedge clk) begin
    r0.v <= drain_go;
    for (int k = 0; k < NL; k++) begin
      r0.m[k] <= drain_go && (32'(k) < 32'(mr));
      r0.nv[k] <= rfh[32*k +: 32];
      r0.tg[k] <= {rfh[MCOLS*32 +: 16], 8'(k)};
    end
    r1 <= r0;
  end
  f32_t ry [NL];
  for (genvar k = 0; k < NL; k++) begin : g_rmw
    otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(1'b1), .a(oldr[k]), .b(alr[k]), .c(r1.nv[k]),
                                        .y(ry[k]));
  end
  otpu_delay #(.W($bits(rmw_t)), .N(LM + LA)) u_rw (.clk, .en(1'b1), .d(r1), .q(rw));
  always_ff @(posedge clk) begin
    if (rst_s) begin
      rf_q <= '0; rf_t <= '0; rf_n <= '0; rf_hs <= 1'b0; ncnt <= '0;
    end else begin
      rf_hs <= d_pop;
      if (rf_hs) rf_q <= rf_q + 1'b1;
      if (rf_push) begin
        rf_t <= rf_t + 1'b1;
        ncnt <= ncnt + 16'd1;
      end
      rf_n <= rf_n + (RFW+1)'(rf_push) - (RFW+1)'(d_pop);
    end
  end
  assign busy = (rf_n != 0);
  always_ff @(posedge clk)
    for (int k = 0; k < NL; k++) begin
      y[k] <= ry[k];
      yv[k] <= rw.v && rw.m[k];
      ytag[k] <= rw.tg[k];
    end
endmodule
