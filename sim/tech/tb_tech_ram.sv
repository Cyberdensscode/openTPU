// The technology RAMs (rtl/tech/<family>/otpu_tech_ram.sv) under random traffic: every cycle
// after the first read prints the read data. Built once with rtl/tech/generic (inferred) and once
// with rtl/tech/asic (tiled onto the macros' models, tools/asic/fakeram.py); tests/test_tech_ram.py
// compares the two traces. KIND 0 otpu_ram_1r1w, 1 otpu_ram_1r1w_be, 2 otpu_ram_1rw. Every word is
// written first (the macros' models start undefined), and a 1R1W read never names the word
// written at the same edge (undefined on a macro; no user of the RAMs does it).
module tb_tech_ram #(parameter int W = 40, parameter int N = 8192, parameter int KIND = 0);
  localparam int A = $clog2(N);
  logic clk = 1'b0;
  logic we, re, en;
  logic [W/8 > 0 ? W/8-1 : 0:0] wbe;
  logic [A-1:0] wa, ra;
  logic [W-1:0] wd, rd;
  integer seed, cyc;
  bit any_read;

  if (KIND == 0) begin : g_2p
    otpu_ram_1r1w #(.W(W), .N(N)) u (.clk, .we, .wa, .wd, .re, .ra, .rd);
  end else if (KIND == 1) begin : g_be
    otpu_ram_1r1w_be #(.W(W), .N(N)) u (.clk, .we, .wbe, .wa, .wd, .re, .ra, .rd);
  end else begin : g_1p
    otpu_ram_1rw #(.W(W), .N(N)) u (.clk, .en, .we, .a(wa), .wd, .rd);
  end

  function automatic logic [W-1:0] rndw();
    logic [W-1:0] x;
    for (int i = 0; i < W; i += 32) x[i +: 32] = $random(seed);
    return x;
  endfunction

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    any_read = 1'b0;
    // fill: every word once (all bytes)
    for (int i = 0; i < N; i++) begin
      we = 1'b1; en = 1'b1; re = 1'b0; wbe = '1; wa = A'(i); ra = '0; wd = rndw();
      #1 clk = 1'b1; #1 clk = 1'b0;
    end
    for (cyc = 0; cyc < 20000; cyc++) begin
      we = ($random(seed) & 1) != 0;
      wa = A'($random(seed));
      wd = rndw();
      wbe = $random(seed);
      if (KIND == 2) begin
        en = ($random(seed) % 4) != 0;
        if (en && !we) any_read = 1'b1;
      end else begin
        re = ($random(seed) % 5) < 3;
        ra = A'($random(seed));
        if (re && we && ra == wa) ra = ra + 1'b1;
        if (re) any_read = 1'b1;
      end
      #1 clk = 1'b1; #1 clk = 1'b0;
      if (any_read) $display("%0d %h", cyc, rd);
    end
    $finish;
  end
endmodule
