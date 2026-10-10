// The MXU's arithmetic cells (rtl/tech/<family>/otpu_tech_dsp.sv) under random stimulus: every
// cycle prints each otpu_pe variant's p and pcout (low 24 bits: all the chain uses) and
// otpu_colend's s. Built once with rtl/tech/generic and once with a vendor family (and its DSP
// model); tests/test_tech.py compares the two traces.
module tb_tech_dsp;
  logic clk = 1'b0;
  logic en;
  logic [7:0] act, wlo, whi;
  logic sel;
  logic [47:0] pcin;
  logic [3:0][23:0] a, d;
  logic [15:0] mb;
  integer seed, cyc;

  // the eight otpu_pe variants: FIRST, LAST, WREG
  logic [7:0][47:0] pco;
  logic signed [7:0][23:0] po;
  for (genvar v = 0; v < 8; v++) begin : g_pe
    otpu_pe #(.FIRST(v[0]), .LAST(v[1]), .WREG(v[2])) u (
      .clk, .en, .act, .wlo, .whi, .sel, .pcin, .pcout(pco[v]), .p(po[v]));
  end
  logic signed [22:0] s;
  otpu_colend #(.SW(23)) u_ce (.clk, .en, .a, .d, .mb, .s);

  // a random signed 24-bit chain sum, sign-extended (the cascade's input)
  function automatic logic [47:0] rnd48();
    logic [23:0] x;
    x = $random(seed);
    return {{24{x[23]}}, x};
  endfunction

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    for (cyc = 0; cyc < 4000; cyc++) begin
      en = (cyc < 16) ? 1'b1 : ($random(seed) % 8 != 0);
      act = $random(seed); wlo = $random(seed); whi = $random(seed); sel = $random(seed);
      pcin = rnd48();
      for (int b = 0; b < 4; b++) begin
        a[b] = $random(seed) % (1 << 20);
        d[b] = $random(seed) % (1 << 20);
      end
      mb = $random(seed);
      #1 clk = 1'b1;
      #1 clk = 1'b0;
      if (cyc >= 16) begin                     // the pipelines filled with defined values
        $write("%0d", cyc);
        for (int v = 0; v < 8; v++)
          $write(" %06x %06x", po[v][23:0], v[1] ? 24'd0 : pco[v][23:0]);
        $write(" %06x\n", {1'b0, s[22:0]});
      end
    end
    $finish;
  end
endmodule
