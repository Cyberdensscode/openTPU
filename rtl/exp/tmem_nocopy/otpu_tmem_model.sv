// EXPERIMENTAL, simulation only (docs/tmem_nocopy.md): the read arbitration of candidate TMEMs
// with one or two copies, for co-simulation (otpu_slice TMEM_IMPL 2). The words themselves stay
// in otpu_tmem IMPL 0 with a copy per read port, so any grant pattern reads the right data; this
// module only decides which units are granted each cycle, i.e. the stalls a candidate costs.
// Units are granted all-or-nothing in the slice's priority order (DMA, COLL, VPU, MXU, QUANT),
// as by every other arbiter of the slice. Writes keep today's rule (one write per bank and
// cycle, the board's), so only the reads differ.
//
// A candidate (plusargs, read at time 0; the defaults in brackets):
//   +tm_nsb=N   sub-banks per bank [8]: each bank (word address mod LANES) is N RAMs, each with
//               its own address; the sub-bank of word a is (a >> sbp) mod N
//   +tm_sbp=B   the sub-bank field's lowest address bit [3]: 3 interleaves 8-word groups, 13
//               (64K words, N 8) gives 8K-word regions
//   +tm_sbh=1   XOR the next field above into the sub-bank (a hash) [0]; 2: XOR every field above
//   +tm_pskew=S each port's addresses seen 8 * S[p] words later for the sub-bank only, one digit
//               per port (port order as tm_cpy): an operand placement that keeps one unit's
//               operands in different sub-banks, the most a compiler's placement could do [none]
//   +tm_rps=R   reads per sub-bank RAM and cycle [1] (2: a true dual-port RAM's second port, or
//               a copy pumped at 2x)
//   +tm_rq=1    a RAM's address is the first asking port's whether or not its unit is granted
//               (no grant on the RAM address path, as TMEM_RQ) [0]
//   +tm_cpy=S   each read port's copy, one digit per port in port order (MXU DMA Q VA VB Q3 Q2
//               COLL); 9: no RAM read (a register in the unit) [00000000]
//   +tm_nbuf=K  lines (8 aligned words) a port keeps of what it read [0]; a read whose lanes all
//               hit them needs no RAM read; a write to a line drops it from every port's lines
//   +tm_bufm=H  the ports with lines, a hex mask by port index [ff]
//   +tm_share=0 two ports reading the same word still take two reads (no address compare) [1]
//   +tm_tdp=1   true dual-port RAMs: a write takes one of the RAM's tm_rps accesses (one copy)
//   +tm_wcpy=K  tdp: the writes take their access on "copy" K's RAMs [0]; with tm_cpy splitting
//               the ports over copies 0 and 1 and tm_rps 1, one true dual-port copy whose port B
//               serves copy 0's ports and port A copy 1's and the writes
// Two reads of the same word share a RAM read (+tm_share=1). A unit whose own reads need more than R reads of
// one RAM (two of its ports in one sub-bank, different words) reads them over several cycles: it
// is granted after ceil(n / R) - 1 more cycles (a holding register in the unit).
module otpu_tmem_model #(
  parameter int LANES = 8,
  parameter int NRP   = 8,
  parameter int NWP   = 4,
  parameter int NG    = 5,
  parameter logic [NRP-1:0][2:0] RGRP = '0,   // each read port's group (group 0 first)
  parameter logic [NWP-1:0][2:0] WGRP = '0,
  parameter int GCOLL = 1,                    // the collective's group (every slice grants it)
  parameter int SID   = 0
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic                            run,      // count this cycle
  input  logic [NRP-1:0][LANES-1:0]       rq_en,
  input  logic [NRP-1:0][LANES-1:0][31:0] r_addr,
  input  logic [NWP-1:0][LANES-1:0]       wq_en,
  input  logic [NWP-1:0][LANES-1:0][31:0] w_addr,
  input  logic [NRP-1:0][LANES-1:0]       r_take,   // the reads that happened (r_en)
  input  logic [NWP-1:0][LANES-1:0]       w_take,   // the writes that happened (w_en)
  input  logic [NG-1:0]                   blk,      // groups held outside the TMEM
  input  logic                            coll_gnt,
  output logic [NG-1:0]                   gnt,
  output logic                            cgl       // the collective may go (this slice)
);
  localparam int BW = $clog2(LANES);
  localparam int MC = 8, MSB = 16, MR = 4, MB = 8;   // copies, sub-banks, reads, lines
  int nsb, lsb, sbp, sbh, rps, rq, nbuf, share, tdp, wcpy;
  int cpy [NRP], psk [NRP];
  logic [NRP-1:0] bufm;
  initial begin
    string s;
    nsb = 8; sbp = 3; sbh = 0; rps = 1; rq = 0; nbuf = 0; bufm = '1; share = 1; tdp = 0; wcpy = 0;
    void'($value$plusargs("tm_nsb=%d", nsb));
    void'($value$plusargs("tm_sbp=%d", sbp));
    void'($value$plusargs("tm_sbh=%d", sbh));
    void'($value$plusargs("tm_rps=%d", rps));
    void'($value$plusargs("tm_rq=%d", rq));
    void'($value$plusargs("tm_nbuf=%d", nbuf));
    void'($value$plusargs("tm_bufm=%h", bufm));
    void'($value$plusargs("tm_share=%d", share));
    void'($value$plusargs("tm_tdp=%d", tdp));
    void'($value$plusargs("tm_wcpy=%d", wcpy));
    for (int p = 0; p < NRP; p++) begin cpy[p] = 0; psk[p] = 0; end
    if ($value$plusargs("tm_cpy=%s", s))
      for (int p = 0; p < NRP && p < s.len(); p++) cpy[p] = int'(s[p]) - 48;
    if ($value$plusargs("tm_pskew=%s", s))
      for (int p = 0; p < NRP && p < s.len(); p++) psk[p] = 8 * (int'(s[p]) - 48);
    lsb = $clog2(nsb);
    if (nsb < 1 || nsb > MSB || (1 << lsb) != nsb || rps < 1 || rps > MR || nbuf > MB)
      $fatal(1, "otpu_tmem_model: nsb %0d (1..%0d, a power of 2), rps %0d (1..%0d), nbuf %0d (..%0d)",
             nsb, MSB, rps, MR, nbuf, MB);
    for (int p = 0; p < NRP; p++)
      if (cpy[p] != 9 && (cpy[p] < 0 || cpy[p] >= MC))
        $fatal(1, "otpu_tmem_model: port %0d's copy %0d (0..%0d or 9)", p, cpy[p], MC - 1);
    $display("TMMODEL cfg sid=%0d nsb=%0d sbp=%0d sbh=%0d rps=%0d rq=%0d nbuf=%0d bufm=%h share=%0d tdp=%0d wcpy=%0d cpy=%0d%0d%0d%0d%0d%0d%0d%0d",
             SID, nsb, sbp, sbh, rps, rq, nbuf, bufm, share, tdp, wcpy, cpy[0], cpy[1], cpy[2], cpy[3], cpy[4],
             cpy[5], cpy[6], cpy[7]);
  end

  function automatic int sbank(input logic [31:0] a);
    logic [31:0] x;
    x = a >> sbp;
    if (sbh == 1) x = x ^ (a >> (sbp + lsb));
    if (sbh == 2 && lsb > 0) for (int f = sbp + lsb; f < 32; f += lsb) x = x ^ (a >> f);
    return int'(x & 32'(nsb - 1));
  endfunction

  // each port's lines: tags (word address >> 3), valid, last use
  logic [31:0] bt [NRP][MB];
  logic        bv [NRP][MB];
  longint      bu [NRP][MB];
  longint      now;
  function automatic logic bhit(input int p, input logic [31:0] a);
    if (nbuf == 0 || !bufm[p]) return 1'b0;
    for (int i = 0; i < nbuf; i++) if (bv[p][i] && bt[p][i] == (a >> 3)) return 1'b1;
    return 1'b0;
  endfunction

  int  ncy [NG];       // cycles a group's own reads need (1: they fit one cycle)
  int  sc  [NG];       // cycles a group has spent on its own reads so far
  logic [NG-1:0] rqg;  // the group asks for reads
  logic [NRP-1:0][LANES-1:0] hitl;   // lanes served by the port's lines
  // The reads the RAMs make this cycle, as a list of {RAM, word}: RAM = copy << 8 | bank << 4 |
  // sub-bank, word = the address >> BW (two reads of one word share)
  localparam int NU = (NRP + NWP) * LANES, NGL = 4 * LANES;
  always_comb begin
    logic [31:0] uk [NU], ur [NU], gk [NGL], gr [NGL];
    int nu, ng, need, rem, nn, used, last;
    logic [LANES-1:0] wt, gw;
    logic ok, dup, first;
    logic [31:0] a, k, r;
    a = '0; k = '0; r = '0; dup = 1'b0; ok = 1'b0; first = 1'b0; gw = '0;
    ng = 0; need = 0; rem = 0; nn = 0; used = 0; last = 0;
    for (int i = 0; i < NU; i++) begin uk[i] = '0; ur[i] = '0; end
    for (int i = 0; i < NGL; i++) begin gk[i] = '0; gr[i] = '0; end
    for (int g = 0; g < NG; g++) ncy[g] = 1;
    nu = 0;
    wt = '0;
    cgl = 1'b1;
    gnt = '0;
    hitl = '0;
    rqg = '0;
    for (int g = 0; g < NG; g++) begin
      // the group's own reads: distinct {RAM, word}
      ng = 0;
      for (int p = 0; p < NRP; p++)
        if (int'(RGRP[p]) == g && cpy[p] != 9)
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l]) begin
              a = r_addr[p][l];
              if (bhit(p, a)) hitl[p][l] = 1'b1;
              else begin
                rqg[g] = 1'b1;
                k = (32'(cpy[p]) << 8) | (32'(a[BW-1:0]) << 4) | 32'(sbank(a + 32'(psk[p])));
                r = (share != 0) ? (a >> BW) : ((a >> BW) | (32'(p) << 28));
                dup = 1'b0;
                for (int m = 0; m < ng; m++) if (gk[m] == k && gr[m] == r) dup = 1'b1;
                if (!dup && ng < NGL) begin gk[ng] = k; gr[ng] = r; ng++; end
              end
            end
      ok = 1'b1;
      // writes: one per bank and cycle (today's rule); tdp: each also takes one of its RAM's
      // accesses (a write never shares)
      gw = '0;
      for (int p = 0; p < NWP; p++)
        if (int'(WGRP[p]) == g)
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) begin
              if (wt[w_addr[p][l][BW-1:0]]) ok = 1'b0;
              gw[w_addr[p][l][BW-1:0]] = 1'b1;
              if (tdp != 0 && ng < NGL) begin
                a = w_addr[p][l];
                gk[ng] = (32'(wcpy) << 8) | (32'(a[BW-1:0]) << 4) | 32'(sbank(a));
                gr[ng] = 32'h8000_0000 | (a >> BW);
                ng++;
              end
            end
      // reads, per RAM (at its first entry in the list): alone the group needs ceil(n / rps)
      // cycles for its n words there; in its last cycle its last words that the RAM does not
      // already read for an earlier group must fit what is left of the RAM's rps reads
      for (int m = 0; m < ng; m++) begin
          first = 1'b1;
          for (int q = 0; q < m; q++) if (gk[q] == gk[m]) first = 1'b0;
          if (first) begin
            need = 0;
            for (int q = 0; q < ng; q++) if (gk[q] == gk[m]) need++;
            if ((need + rps - 1) / rps > ncy[g]) ncy[g] = (need + rps - 1) / rps;
            rem = need - rps * ((need + rps - 1) / rps - 1);
            used = 0;
            for (int q = 0; q < nu; q++) if (uk[q] == gk[m]) used++;
            nn = 0;
            last = 0;
            for (int q = 0; q < ng; q++)
              if (gk[q] == gk[m]) begin
                last++;
                if (last > need - rem) begin
                  dup = 1'b0;
                  for (int t = 0; t < nu; t++) if (uk[t] == gk[q] && ur[t] == gr[q]) dup = 1'b1;
                  if (!dup) nn++;
                end
              end
            if (used + nn > rps) ok = 1'b0;
          end
        end
      if (ncy[g] > 1 && sc[g] < ncy[g] - 1) ok = 1'b0;
      if (blk[g]) ok = 1'b0;
      if (g == GCOLL) begin
        cgl = ok;
        gnt[g] = coll_gnt;
      end else begin
        gnt[g] = ok;
      end
      // what the RAMs read: the granted group's words (rq: an asking group's, granted or not)
      if (ok || rq != 0)
        for (int m = 0; m < ng; m++) begin
            dup = 1'b0;
            for (int t = 0; t < nu; t++) if (uk[t] == gk[m] && ur[t] == gr[m]) dup = 1'b1;
            if (!dup && nu < NU) begin uk[nu] = gk[m]; ur[nu] = gr[m]; nu++; end
          end
      if (ok) wt = wt | gw;
    end
  end

  // the lines and the serialization counters; statistics (blocking: a sequential algorithm)
  /* verilator lint_off BLKSEQ */
  longint st_cyc, st_lose [NG], st_self [NG], st_look [NRP], st_hit [NRP];
  initial begin
    now = 0; st_cyc = 0;
    for (int g = 0; g < NG; g++) begin st_lose[g] = 0; st_self[g] = 0; sc[g] = 0; end
    for (int p = 0; p < NRP; p++) begin
      st_look[p] = 0; st_hit[p] = 0;
      for (int i = 0; i < MB; i++) begin bv[p][i] = 1'b0; bt[p][i] = '0; bu[p][i] = 0; end
    end
  end
  always @(posedge clk) begin
    now <= now + 1;
    if (rst) begin
      for (int g = 0; g < NG; g++) sc[g] <= 0;
      for (int p = 0; p < NRP; p++) for (int i = 0; i < MB; i++) bv[p][i] <= 1'b0;
    end else begin
      for (int g = 0; g < NG; g++)
        sc[g] <= (rqg[g] && ncy[g] > 1 && !gnt[g]) ? sc[g] + 1 : 0;
      if (nbuf > 0) begin
        // a write drops its line from every port's lines (it lands in the RAM next cycle)
        for (int p = 0; p < NWP; p++)
          for (int l = 0; l < LANES; l++)
            if (w_take[p][l])
              for (int q = 0; q < NRP; q++)
                for (int i = 0; i < nbuf; i++)
                  if (bt[q][i] == (w_addr[p][l] >> 3)) bv[q][i] = 1'b0;
        // the lines a port read come in (the least recently used goes)
        for (int p = 0; p < NRP; p++)
          if (bufm[p])
            for (int l = 0; l < LANES; l++)
              if (r_take[p][l]) begin
                logic [31:0] t;
                int v;
                logic found;
                t = r_addr[p][l] >> 3;
                found = 1'b0;
                for (int i = 0; i < nbuf; i++)
                  if (bv[p][i] && bt[p][i] == t) begin found = 1'b1; bu[p][i] = now; end
                if (!found) begin
                  v = 0;
                  for (int i = 0; i < nbuf; i++)
                    if (!bv[p][i]) begin v = i; break; end
                    else if (bu[p][i] < bu[p][v]) v = i;
                  bt[p][v] = t; bv[p][v] = 1'b1; bu[p][v] = now;
                end
              end
      end
      if (run) begin
        st_cyc++;
        for (int g = 0; g < NG; g++) begin
          if (rqg[g] && !gnt[g]) st_lose[g]++;
          if (rqg[g] && ncy[g] > 1 && !gnt[g]) st_self[g]++;
        end
        for (int p = 0; p < NRP; p++)
          for (int l = 0; l < LANES; l++)
            if (r_take[p][l]) begin
              st_look[p]++;
              if (hitl[p][l]) st_hit[p]++;
            end
      end
    end
  end
  /* verilator lint_on BLKSEQ */
  final begin
    string s;
    s = $sformatf("TMMODEL sid=%0d cyc=%0d lose=", SID, st_cyc);
    for (int g = 0; g < NG; g++) s = {s, $sformatf("%s%0d", (g > 0) ? "," : "", st_lose[g])};
    s = {s, " self="};
    for (int g = 0; g < NG; g++) s = {s, $sformatf("%s%0d", (g > 0) ? "," : "", st_self[g])};
    s = {s, " look="};
    for (int p = 0; p < NRP; p++) s = {s, $sformatf("%s%0d", (p > 0) ? "," : "", st_look[p])};
    s = {s, " hit="};
    for (int p = 0; p < NRP; p++) s = {s, $sformatf("%s%0d", (p > 0) ? "," : "", st_hit[p])};
    $display("%s", s);
  end
endmodule
