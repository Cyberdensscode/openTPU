// The technology RAMs (rtl/tech/<family>/otpu_tech_ram.sv) under random traffic: every cycle
// after the first read prints the read data. Built once with rtl/tech/generic (inferred) and once
// with rtl/tech/asic (tiled onto the macros' models, tools/asic/fakeram.py); tests/test_tech_ram.py
// compares the two traces. KIND 0 otpu_ram_1r1w, 1 otpu_ram_1r1w_be, 2 otpu_ram_1rw, 3
// otpu_ram_1rw1r, 4 otpu_ram_1r1w_ra (read ahead: its reads name any word, the one written at
// the same edge included). Every word is written first (the macros' models start undefined), and a 1R1W
// read never names the word written at the same edge (undefined on a macro; no user of the RAMs
// does it). The 1RW1R RAM reads first on both ports, so its reads name any word: port A's write
// cycles and port B's reads of the word written print the word before the write.
module tb_tech_ram #(parameter int W = 40, parameter int N = 8192, parameter int KIND = 0);
  localparam int A = $clog2(N);
  logic clk = 1'b0;
  logic we, re, en;
  logic [W/8 > 0 ? W/8-1 : 0:0] wbe;
  logic [A-1:0] wa, ra;
  logic [W-1:0] wd, rd, rdb;
  integer seed, cyc;
  bit any_read, any_b;

  if (KIND == 0) begin : g_2p
    otpu_ram_1r1w #(.W(W), .N(N)) u (.clk, .we, .wa, .wd, .re, .ra, .rd);
  end else if (KIND == 1) begin : g_be
    otpu_ram_1r1w_be #(.W(W), .N(N)) u (.clk, .we, .wbe, .wa, .wd, .re, .ra, .rd);
  end else if (KIND == 4) begin : g_ra
    otpu_ram_1r1w_ra #(.W(W), .N(N)) u (.clk, .we, .wa, .wd, .ra_n(ra), .rd);
  end else if (KIND == 2) begin : g_1p
    otpu_ram_1rw #(.W(W), .N(N)) u (.clk, .en, .we, .a(wa), .wd, .rd);
  end else begin : g_dp
    otpu_ram_1rw1r #(.W(W), .N(N)) u (.clk, .ena(en), .wea(we), .aa(wa), .wda(wd), .rda(rd),
                                      .enb(re), .ab(ra), .rdb);
  end

  function automatic logic [W-1:0] rndw();
    logic [W-1:0] x;
    for (int i = 0; i < W; i += 32) x[i +: 32] = $random(seed);
    return x;
  endfunction

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    any_read = 1'b0;
    any_b = 1'b0;
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
      end else if (KIND == 3) begin
        // port A: a read or a write (read first: rd is the word before a write); port B: a
        // read of any word, the one port A writes included
        en = ($random(seed) % 4) != 0;
        re = ($random(seed) % 5) < 3;
        ra = (($random(seed) % 4) == 0) ? wa : A'($random(seed));
        if (en) any_read = 1'b1;
        if (re) any_b = 1'b1;
      end else if (KIND == 4) begin
        // read ahead: every cycle reads (rd is the word at last cycle's ra_n), a third of them
        // the word written at the same edge (the bypass)
        ra = (($random(seed) % 3) == 0) ? wa : A'($random(seed));
        any_read = 1'b1;
      end else begin
        re = ($random(seed) % 5) < 3;
        ra = A'($random(seed));
        if (re && we && ra == wa) ra = ra + 1'b1;
        if (re) any_read = 1'b1;
      end
      #1 clk = 1'b1; #1 clk = 1'b0;
      if (KIND == 3) begin
        if (any_read || any_b) $display("%0d %h %h", cyc, any_read ? rd : '0, any_b ? rdb : '0);
      end else if (any_read) $display("%0d %h", cyc, rd);
    end
    $finish;
  end
endmodule
