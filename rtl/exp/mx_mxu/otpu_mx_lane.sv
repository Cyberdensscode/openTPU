// EXPERIMENTAL (docs/mx_mxu.md): not in any file list, not in the board build.
//
// A shared epilogue lane of the MX MXU: one row result (a 48-bit accumulator, PAIR's two halves
// already added) a cycle, once per output instead of once per block and column:
//   v  = i2f48(x)            round to nearest even (|x| < 2^47: no overflow, no subnormal)
//   y1 = v * Sw[n]           otpu_fmul
//   y2 = y1 * Sa'[r]         otpu_fmul; Sa' = Sa[r] * 2^-Wf (the window's power of two)
//   y  = old * alpha + y2    otpu_fmadd, as the drain's read-modify-write lanes (ACC: alpha 1)
// Latency LAT (13): i2f 3 | fmul 2 | fmul 2 | fmadd 6. old / alpha are taken at the fmadd
// (the TMEM read of the RMW arrives there); sw at the first multiply, sa at the second.
module otpu_mx_lane
  import otpu_fp::*;
(
  input  logic         clk,
  input  logic         v,
  input  logic [47:0]  x,
  input  f32_t         sw,
  input  f32_t         sa,
  input  f32_t         old,
  input  f32_t         alpha,
  input  logic [23:0]  tag,
  output f32_t         y,
  output logic         yv,
  output logic [23:0]  ytag
);
  localparam int LM = 2, LA = 4;
  localparam int LAT = 3 + 2 * LM + LM + LA;

  function automatic logic [5:0] lzc48(input logic [47:0] m);
    if (m[47:16] != 0) return lzc32(m[47:16]);
    return 6'd32 + 6'(lzc32({m[15:0], 16'd0}));
  endfunction
  // L1: sign, magnitude
  logic        s1, z1;
  logic [47:0] g1;
  // L2: leading zeros
  logic        s2, z2;
  logic [47:0] g2;
  logic [5:0]  lz2;
  // L3: normalize, round
  f32_t        fv;
  always_ff @(posedge clk) begin
    s1 <= x[47];
    z1 <= (x == 0);
    g1 <= x[47] ? (~x + 48'd1) : x;
    s2 <= s1; z2 <= z1; g2 <= g1;
    lz2 <= lzc48(g1);
    begin
      logic [47:0] nrm;
      logic [23:0] mm;
      logic [24:0] mr;
      logic g, st;
      logic [8:0] e;
      nrm = g2 << lz2;
      e = 9'd174 - 9'(lz2);                     // 127 + 47 - lz
      mm = nrm[47:24];
      g = nrm[23];
      st = |nrm[22:0];
      mr = {1'b0, mm} + ((g && (st || mm[0])) ? 25'd1 : 25'd0);
      if (mr[24]) begin
        mr = mr >> 1;
        e = e + 9'd1;
      end
      fv <= z2 ? F_ZERO : {s2, e[7:0], mr[22:0]};
    end
  end
  f32_t swd, sad, y1, y2;
  otpu_delay #(.W(32), .N(3)) u_sw (.clk, .en(1'b1), .d(sw), .q(swd));
  otpu_delay #(.W(32), .N(3 + LM)) u_sa (.clk, .en(1'b1), .d(sa), .q(sad));
  otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en(1'b1), .a(fv), .b(swd), .y(y1));
  otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en(1'b1), .a(y1), .b(sad), .y(y2));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(1'b1), .a(old), .b(alpha), .c(y2), .y);
  otpu_delay #(.W(25), .N(LAT)) u_t (.clk, .en(1'b1), .d({v, tag}), .q({yv, ytag}));
endmodule
