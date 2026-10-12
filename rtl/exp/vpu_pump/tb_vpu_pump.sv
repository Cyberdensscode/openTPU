// EXPERIMENTAL (docs/vpu_pump.md): the pumped multiply-add pair against two of today's units.
// clk_f toggles every time unit, clk_s every two (both rise together); random operands each core
// cycle (with zeros, denormals, infinities and NaNs mixed in) and a random pipeline enable. Each
// design's results are recorded on the enabled core cycles; the pumped streams must equal the
// base's at a fixed offset (their latency difference in enabled cycles) for every lane.
// Run: rtl/exp/vpu_pump/run_tb.sh (Verilator).
module tb_vpu_pump;
  import otpu_fp::*;
  localparam int N = 20000;          // core cycles
  // both rise at 10, 30, 50, ... (clk_s every other time: 10, 50, 90, ...; clk_f's 30, 70, ...
  // are the mid edges)
  logic clk_s = 0, clk_f = 0;
  initial begin #10; forever begin clk_f = 1; #10; clk_f = 0; #10; end end
  initial begin #10; forever begin clk_s = 1; #20; clk_s = 0; #20; end end
  logic en = 0;
  f32_t a0, b0, c0, e0, a1, b1, c1, e1;
  f32_t yb0, yb1, yp0, yp1, yd0, yd1, ys0, ys1;
  vp_pair_base u_b (.clk_s, .en, .a0, .b0, .c0, .e0, .a1, .b1, .c1, .e1, .y0(yb0), .y1(yb1));
  vp_pair_pump #(.DEEP(0)) u_p (.clk_s, .clk_f, .en, .a0, .b0, .c0, .e0, .a1, .b1, .c1, .e1,
                                   .y0(yp0), .y1(yp1));
  vp_pair_pump #(.DEEP(1), .XR(1)) u_d (.clk_s, .clk_f, .en, .a0, .b0, .c0, .e0, .a1, .b1,
                                        .c1, .e1, .y0(yd0), .y1(yd1));
  vp_pair_pump #(.DEEP(2)) u_s (.clk_s, .clk_f, .en, .a0, .b0, .c0, .e0, .a1, .b1, .c1, .e1,
                                .y0(ys0), .y1(ys1));
  function automatic f32_t rnd();
    int k;
    k = $urandom_range(0, 99);
    if (k < 4) return F_ZERO ^ {$urandom_range(0, 1) == 1, 31'd0};      // +-0
    if (k < 7) return {$urandom_range(0, 1) == 1, 8'd0, 23'($urandom)};  // denormal
    if (k < 9) return {$urandom_range(0, 1) == 1, 8'hFF, 23'd0};         // +-inf
    if (k < 10) return F_NAN;
    // normal, exponents near 127 so that products and sums meet (and cancel) often
    return {$urandom_range(0, 1) == 1, 8'($urandom_range(100, 154)), 23'($urandom)};
  endfunction
  f32_t qb0 [$], qb1 [$], qp0 [$], qp1 [$], qd0 [$], qd1 [$], qs0 [$], qs1 [$];
  initial begin
    int bad, off_p, off_d, off_s, warm;
    repeat (2) @(posedge clk_s);
    for (int n = 0; n < N; n++) begin
      // core cycle n: its enable and operands just after its slow edge; its results just
      // before the next slow edge, kept when it is enabled (what a clk_s register with that
      // enable takes at the edge)
      @(posedge clk_s);
      #1;
      en = ($urandom_range(0, 99) < 75);
      a0 = rnd(); b0 = rnd(); c0 = rnd(); e0 = (n % 3 == 0) ? F_ONE : rnd();
      a1 = rnd(); b1 = rnd(); c1 = rnd(); e1 = (n % 5 == 0) ? F_ONE : rnd();
      #37;
      if (en) begin
        qb0.push_back(yb0); qb1.push_back(yb1);
        qp0.push_back(yp0); qp1.push_back(yp1);
        qd0.push_back(yd0); qd1.push_back(yd1);
        qs0.push_back(ys0); qs1.push_back(ys1);
      end
    end
    if ($test$plusargs("dump"))
      for (int i = 100; i < 112; i++)
        $display("%0d base %h %h pump %h %h deep %h %h", i, qb0[i], qb1[i], qp0[i], qp1[i],
                 qd0[i], qd1[i]);
    if ($test$plusargs("dump"))
      for (int i = 40; i < qb0.size() - 20; i++)
        if (qp0[i] != qb0[i + 2] || qp1[i] != qb1[i + 2]) begin
          $display("first pump mismatch at %0d: base %h %h pump %h %h", i, qb0[i+2], qb1[i+2],
                   qp0[i], qp1[i]);
          break;
        end
    warm = 40;
    off_p = -1; off_d = -1; off_s = -1;
    for (int d = -12; d <= 12 && off_p < 0; d++) begin
      int ok;
      ok = 1;
      for (int i = warm; i < qb0.size() - 20; i++)
        if (qp0[i] != qb0[i + d] || qp1[i] != qb1[i + d]) begin ok = 0; break; end
      if (ok) off_p = d + 100;
    end
    for (int d = -12; d <= 12 && off_d < 0; d++) begin
      int ok;
      ok = 1;
      for (int i = warm; i < qb0.size() - 20; i++)
        if (qd0[i] != qb0[i + d] || qd1[i] != qb1[i + d]) begin ok = 0; break; end
      if (ok) off_d = d + 100;
    end
    for (int d = -12; d <= 12 && off_s < 0; d++) begin
      int ok;
      ok = 1;
      for (int i = warm; i < qb0.size() - 20; i++)
        if (qs0[i] != qb0[i + d] || qs1[i] != qb1[i + d]) begin ok = 0; break; end
      if (ok) off_s = d + 100;
    end
    $display("enabled cycles %0d; pumped (today's unit at clk_f): %s, latency %0d enabled core cycles vs the base's 6",
             qb0.size(), off_p >= 0 ? "BIT-EXACT" : "MISMATCH", off_p >= 0 ? 6 - (off_p - 100) : -1);
    $display("pumped deep (XR 1): %s, latency %0d enabled core cycles vs the base's 6",
             off_d >= 0 ? "BIT-EXACT" : "MISMATCH", off_d >= 0 ? 6 - (off_d - 100) : -1);
    $display("pumped split (10 fast stages): %s, latency %0d enabled core cycles vs the base's 6",
             off_s >= 0 ? "BIT-EXACT" : "MISMATCH", off_s >= 0 ? 6 - (off_s - 100) : -1);
    if (off_p < 0 || off_d < 0 || off_s < 0) $fatal(1, "tb_vpu_pump: MISMATCH");
    $display("PASS");
    $finish;
  end
endmodule
