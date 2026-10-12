// EXPERIMENTAL (docs/mx_mxu.md): the out-of-context top of the MX MXU's datapath, from the
// chunk and the ACT rows to the read-modify-write lanes: NCOL physical columns of two stationary
// rows each (ROWS = 2 NCOL, the virtual MCOLS), the rows' 48-bit accumulators, PAIR's add, a
// result FIFO and E shared epilogue lanes. The family file (otpu_mx_x7.sv: pumped DSP48E1s,
// otpu_mx_us.sv: packed DSP48E2s) gives the columns, the front and the clocks. base_top.sv is
// today's datapath with the same surroundings. Every input is registered in clk_s (the ports carry
// no timing), the outputs too; the issuer, the FIFOs of chunks and scales, the TMEM addresses and
// RMAX are not here (the same in both).
//
// Chunk t (its inputs in cycle t at the ports): the 128-byte chunk (wf: 0 int8, 1 int4, 2 E2M1;
// h: a 4-bit chunk's high half without PAIR), its two scale words (low / high stream: dw [7:0],
// m_b [8+6b +: 6]), the rows' activation blocks and their exponent words (da: 4 x 8 bits), the
// row flags (v, first, last) and, with the row's last chunk, Sw[n]. Per command: hsel (a physical
// column's stream: PAIR puts the low rows in columns 0 .. NCOL/2-1 and the high rows in the rest,
// so row j's partner is row j + ROWS/2), pair, fmt, m (results per row), Sa[r], old / alpha (the
// RMW's operands, from TMEM in the MXU).
module mx_top
  import otpu_fp::*;
  import otpu_mx_fam::*;
#(
  parameter int NCOL = 2,
  parameter int E = 2,
  parameter int D = 128,
  parameter int RF = 64
) (
  input  logic                          clk_in,
  input  logic                          rst,
  input  logic [2*NCOL-1:0][D-1:0][7:0] x_in,
  input  logic [D-1:0][7:0]             chunk_in,
  input  logic [1:0]                    wf_in,
  input  logic                          h_in,
  input  logic [31:0]                   sw_lo_in,
  input  logic [31:0]                   sw_hi_in,
  input  logic [2*NCOL-1:0][31:0]       da_in,
  input  logic [NCOL-1:0]               hsel_in,
  input  logic                          pair_in,
  input  logic                          fmt_in,
  input  logic [7:0]                    m_in,
  input  logic                          v_in,
  input  logic                          first_in,
  input  logic                          last_in,
  input  logic [31:0]                   swr_in,
  input  logic [2*NCOL-1:0][31:0]       sa_in,
  input  logic [E-1:0][31:0]            old_in,
  input  logic [E-1:0][31:0]            alpha_in,
  output logic [E-1:0][31:0]            y,
  output logic [E-1:0]                  yv,
  output logic [E-1:0][23:0]            ytag,
  output logic                          busy
);
  localparam int ROWS = 2 * NCOL;
  localparam int RFW = $clog2(RF);
  initial if (NCOL < 2 || ROWS % E != 0) $fatal(1, "mx_top: NCOL >= 2, E divides 2 NCOL");

  logic clk_s, clk_f, locked;
  otpu_mx_clk u_clk (.clk_in, .clk_s, .clk_f, .locked);
  logic [1:0] rs;
  always_ff @(posedge clk_s) rs <= {rs[0], rst | ~locked};
  wire rst_s = rs[1];

  // ---- R: the inputs, registered
  logic [ROWS-1:0][D-1:0][7:0] xr;
  logic [D-1:0][7:0] cr;
  logic [1:0] wfr;
  logic hr, pairr, fmtr, vr, fr, lr;
  logic [31:0] swlr, swhr, swrr;
  logic [ROWS-1:0][31:0] dar, sar;
  logic [NCOL-1:0] hselr;
  logic [7:0] mr;
  logic [E-1:0][31:0] oldr, alr;
  always_ff @(posedge clk_s) begin
    xr <= x_in; cr <= chunk_in; wfr <= wf_in; hr <= h_in; swlr <= sw_lo_in; swhr <= sw_hi_in;
    dar <= da_in; hselr <= hsel_in; pairr <= pair_in; fmtr <= fmt_in; mr <= m_in;
    vr <= v_in; fr <= first_in; lr <= last_in; swrr <= swr_in; sar <= sa_in;
    oldr <= old_in; alr <= alpha_in;
  end

  // ---- S1: the chunk decoded (otpu_mxu's wd), everything else a cycle later alongside
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
  logic [D-1:0][15:0] wd;
  logic [ROWS-1:0][D-1:0][7:0] xd;
  logic [31:0] swl1, swh1;
  logic [ROWS-1:0][31:0] da1;
  logic v1, f1, l1;
  always_ff @(posedge clk_s) begin
    for (int i = 0; i < D; i++) wd[i] <= {wsel(cr, i, 1'b1, wfr), wsel(cr, i, hr, wfr)};
    xd <= xr; swl1 <= swlr; swh1 <= swhr; da1 <= dar; v1 <= vr; f1 <= fr; l1 <= lr;
  end

  // ---- the columns
  logic [NCOL:0][D-1:0][15:0] wcs;
  otpu_mx_front #(.D(D)) u_front (.clk_s, .w(wd), .wc(wcs[0]));
  logic [ROWS-1:0][47:0] acc;                 // aligned: chunk t's row in t + LATC + NCOL - 1
  logic [NCOL-1:0] avc;
  for (genvar p = 0; p < NCOL; p++) begin : g_col
    logic [47:0] a0, a1;
    logic av;
    otpu_mx_col2 #(.D(D), .COL(p)) u_col (
      .clk_s, .clk_f, .rst_s, .x0(xd[2*p]), .x1(xd[2*p+1]), .wc_prev(wcs[p]), .wc(wcs[p+1]),
      .hsel(hselr[p]), .sw_lo(swl1), .sw_hi(swh1), .da0(da1[2*p]), .da1(da1[2*p+1]),
      .fmt(fmtr), .v(v1), .first(f1), .last(l1), .acc0(a0), .acc1(a1), .av);
    otpu_skew #(.W(97), .N(NCOL - 1 - p)) u_dsk (.clk(clk_s), .en(1'b1), .d({av, a1, a0}),
                                                .q({avc[p], acc[2*p+1], acc[2*p]}));
  end
  // Sw[n] alongside (from R: one cycle more than the columns' inputs)
  logic [31:0] swa;
  otpu_delay #(.W(32), .N(1 + LATC + NCOL - 1)) u_swd (.clk(clk_s), .en(1'b1), .d(swrr), .q(swa));

  // ---- PAIR's add (row j + its partner j + ROWS/2), the row registered for the FIFO
  logic [ROWS-1:0][47:0] rowp;
  logic rv;
  logic [31:0] swp;
  always_ff @(posedge clk_s) begin
    for (int j = 0; j < ROWS; j++)
      rowp[j] <= (pairr && j < ROWS / 2) ? acc[j] + acc[j + ROWS / 2] : acc[j];
    rv <= avc[0];
    swp <= swa;
  end

  // ---- result FIFO (otpu_mxu's: rows in LUT RAM, three read copies, the head registered)
  localparam int FW = ROWS * 48 + 32 + 16;
  typedef logic [FW-1:0] frow_t;
  logic [RFW-1:0] rf_q, rf_t;
  logic [RFW:0]   rf_n;
  logic           rf_hs;
  logic [15:0]    ncnt;
  frow_t          fin, rh0, rh1, rfh;
  logic [2:0][FW-1:0] rq;
  logic [2:0][RFW-1:0] rf_qc;
  wire rf_push = rv;
  always_comb begin
    fin = {ncnt, swp, rowp};
    for (int c = 0; c < 3; c++) rf_qc[c] = rf_q + RFW'(c);
  end
  for (genvar c = 0; c < 3; c++) begin : g_rfm
    (* ram_style = "distributed" *) logic [FW-1:0] m [RF];
    always_ff @(posedge clk_s) if (rf_push) m[rf_t] <= fin;
    assign rq[c] = (rf_push && rf_t == rf_qc[c]) ? fin : m[rf_qc[c]];
  end
  always_ff @(posedge clk_s) begin
    rh0 <= rf_hs ? rq[1] : rq[0];
    rh1 <= rf_hs ? rq[2] : rq[1];
  end
  assign rfh = rf_hs ? rh1 : rh0;

  // ---- drain: E results of the head row a cycle into the lanes
  localparam int DW = $clog2(ROWS) + 1;
  logic [DW-1:0] dj;
  wire drain_go = (rf_n != 0);
  wire d_fin = 32'(dj) + E >= 32'(mr);
  wire d_pop = drain_go && d_fin;
  // Sa' = Sa * 2^-Wf, registered per command
  logic [ROWS-1:0][31:0] sap;
  always_ff @(posedge clk_s)
    for (int r = 0; r < ROWS; r++) begin
      logic [8:0] e;
      e = 9'(sar[r][30:23]) - (fmtr ? 9'd16 : 9'd11);
      sap[r] <= (e[8] || e == 0) ? {sar[r][31], 31'd0} : {sar[r][31], e[7:0], sar[r][22:0]};
    end
  logic [E-1:0] lv;
  logic [E-1:0][47:0] lx;
  logic [E-1:0][31:0] lsw, lsa;
  logic [E-1:0][23:0] ltg;
  always_ff @(posedge clk_s) begin
    for (int k = 0; k < E; k++) begin
      logic [DW-1:0] j;
      j = dj + DW'(k);
      lv[k] <= drain_go && (32'(j) < 32'(mr));
      lx[k] <= rfh[48 * j +: 48];
      lsw[k] <= rfh[ROWS * 48 +: 32];
      lsa[k] <= sap[j];
      ltg[k] <= {rfh[ROWS * 48 + 32 +: 16], 8'(j)};
    end
  end
  always_ff @(posedge clk_s) begin
    if (rst_s) begin
      rf_q <= '0; rf_t <= '0; rf_n <= '0; rf_hs <= 1'b0; dj <= '0; ncnt <= '0;
    end else begin
      rf_hs <= d_pop;
      if (rf_hs) rf_q <= rf_q + 1'b1;
      if (rf_push) begin
        rf_t <= rf_t + 1'b1;
        ncnt <= ncnt + 16'd1;
      end
      rf_n <= rf_n + (RFW+1)'(rf_push) - (RFW+1)'(d_pop);
      if (drain_go) dj <= d_fin ? '0 : dj + DW'(E);
    end
  end
  assign busy = (rf_n != 0);

  // ---- the lanes
  for (genvar k = 0; k < E; k++) begin : g_lane
    f32_t ly;
    logic lyv;
    logic [23:0] lyt;
    otpu_mx_lane u_l (.clk(clk_s), .v(lv[k]), .x(lx[k]), .sw(lsw[k]), .sa(lsa[k]),
                      .old(oldr[k]), .alpha(alr[k]), .tag(ltg[k]), .y(ly), .yv(lyv), .ytag(lyt));
    always_ff @(posedge clk_s) begin
      y[k] <= ly;
      yv[k] <= lyv;
      ytag[k] <= lyt;
    end
  end
endmodule
