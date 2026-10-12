// EXPERIMENTAL (docs/tmem_pump.md; in no file list): otpu_tmem with its block RAM copies pumped
// at twice the core clock. clk_f = 2 x clk (the same MMCM, both at phase 0: every core edge is a
// fast edge). Each pumped copy is LANES banks of one true dual-port RAM on clk_f; per core cycle
// and bank it serves:
//   - at the core edge (an even fast edge): two reads, port A and port B, from the read ports'
//     requests of the cycle that ends there -- the same edge otpu_tmem's copies read at;
//   - at the fast edge in the middle of the core cycle (odd): port A writes the registered write
//     stage's word (pw_*), which otpu_tmem lands at the next core edge.
// So one pumped copy holds two read slots (A, B) and the write: NS slots need NPC = NS / 2 copies.
// A read's data is a RAM output that holds for the whole next core cycle (port A's write is
// NO_CHANGE, port B only reads), as otpu_tmem's do: the units see the same cycle. A third read
// per copy cannot keep that: it would be issued half a cycle early (before its request settles)
// or its data would land half a cycle late; see docs/tmem_pump.md.
//
// Behaviour, port for port, is otpu_tmem's with WPB = 1:
//   IMPL 0: a slot per read port's copy (the hosts: every port but the guests SH_MASK, which share
//           SH_HOST's slot in priority order), reads from the requests: 6 slots for the slice's 8
//           ports, 3 pumped copies instead of 6.
//   IMPL 1: banked, a slot per CPY value (NCP slots: the "copies" of otpu_tmem IMPL 1, two to a
//           pumped copy), a bank's slot address from the port granted there (r_en).
// The writes, the registered write stage, the bypass of a read that meets the landing write, and
// the lanes' data path are otpu_tmem's code. A write lands half a core cycle earlier than in
// otpu_tmem (the odd edge before the core edge), so a read at that core edge sees it in the RAM;
// the bypass returns the same word either way (it compares against the landing write, pw_*), so
// the read data are bit for bit otpu_tmem's. Reads (even edges) and writes (odd edges) never meet
// in one fast cycle, so the RAM's collision behaviour does not matter. And since the landing write
// is always in the RAM before the read, the bypass is not needed at all (BYP 0, the default: no
// pd registers, no address compare, no multiplexer between the RAM and the lanes; the testbench
// passes either way).
//
// The fast phase: per bank a toggle in the core domain (tog) and its copy in the fast domain (tq):
// they differ during the first half of a core cycle, so the next fast edge is odd; no reset (it
// aligns itself from the first core edge). Timing contract (rtl/exp/tmem_pump/ooc.tcl): the read
// address / enable inputs of a bank (u_pb ra / rena / rb / renb) are sampled on even fast edges
// only (a core-period multicycle path from clk); the RAM outputs change only after even edges
// (a core-period multicycle path into clk); the write stage and the toggle reach the RAM in one
// fast period.
module otpu_tmem_pump #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 8,
  parameter int NWP   = 4,
  parameter logic [NRP-1:0] GEN_R = '1,
  parameter logic [NWP-1:0] GEN_W = '1,
  parameter int SH_HOST = 0,
  parameter logic [NRP-1:0] SH_MASK = '0,
  parameter int IMPL  = 0,
  parameter int NCP   = 1,                 // IMPL 1: slots
  parameter logic [NRP-1:0][2:0] CPY = '0, // IMPL 1: each read port's slot (< NCP)
  parameter bit    RQ    = 1'b0,           // IMPL 1: slot addresses from the requests by PRANK
                                           // (otpu_tmem RQ; the slice's TMEM_RQ arbiter)
  parameter logic [NRP-1:0][2:0] PRANK = '0,
  parameter string STYLE = "block",        // the banks' ram_style ("block", "ultra")
  parameter bit    BYP   = 1'b0,           // keep otpu_tmem's bypass (redundant here: see below)
  parameter bit    FW    = 1'b1            // the write stage and the phase in the fast domain
                                           // (below; 0: both in the core domain, as first built)
) (
  input  logic                                clk,
  input  logic                                clk_f,
  input  logic [NRP-1:0][LANES-1:0]           r_en,
  input  logic [NRP-1:0][LANES-1:0]           r_req,
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,
  input  logic [NWP-1:0][LANES-1:0]           w_req,
  input  logic [NWP-1:0]                      w_gnt,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_addr,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_data
);
  localparam int BW = $clog2(LANES);
  localparam int BD = WORDS / LANES;
  localparam int IW = $clog2(BD);

  function automatic int host_of(input int p);
    if (IMPL == 0) return SH_MASK[p] ? SH_HOST : p;
    for (int q = 0; q < NRP; q++) if (CPY[q] == CPY[p]) return q;
    return p;
  endfunction
  // each host's slot: IMPL 0 the hosts in port order, IMPL 1 its CPY value
  function automatic int slot_of(input int h);
    int n;
    if (IMPL == 1) return int'(CPY[h]);
    n = 0;
    for (int q = 0; q < h; q++) if (host_of(q) == q) n++;
    return n;
  endfunction
  function automatic int n_slots();
    int n;
    if (IMPL == 1) return NCP;
    n = 0;
    for (int q = 0; q < NRP; q++) if (host_of(q) == q) n++;
    return n;
  endfunction
  // the host on slot s (-1: none)
  function automatic int host_on(input int s);
    for (int q = 0; q < NRP; q++) if (host_of(q) == q && slot_of(q) == s) return q;
    return -1;
  endfunction
  localparam int NS  = n_slots();
  localparam int NPC = (NS + 1) / 2;
  initial if (IMPL == 1) for (int p = 0; p < NRP; p++)
    if (CPY[p] >= NCP) $fatal(1, "otpu_tmem_pump: port %0d's slot %0d >= NCP %0d", p, CPY[p], NCP);

  // ---- write selection (otpu_tmem g_w1, WPB = 1)
  logic [LANES-1:0]         bw_v;
  logic [LANES-1:0][IW-1:0] bw_a;
  logic [LANES-1:0][31:0]   bw_d;
  logic [NWP-1:0][LANES-1:0]         pv;
  logic [NWP-1:0][LANES-1:0][IW-1:0] pa;
  logic [NWP-1:0][LANES-1:0][31:0]   pdt;
  for (genvar p = 0; p < NWP; p++) begin : g_wp
    if (GEN_W[p]) begin : g_x
      always_comb begin
        pv[p] = '0; pa[p] = '0; pdt[p] = '0;
        for (int b = 0; b < LANES; b++)
          for (int l = 0; l < LANES; l++)
            if (w_req[p][l] && w_addr[p][l][BW-1:0] == BW'(b)) begin
              pv[p][b] = 1'b1;
              pa[p][b] = pa[p][b] | w_addr[p][l][BW +: IW];
              pdt[p][b] = pdt[p][b] | w_data[p][l];
            end
      end
    end else begin : g_r
      wire [BW+IW-1:0] a0 = w_addr[p][0][BW+IW-1:0];
      wire [BW-1:0]    rot = a0[BW-1:0];
      always_comb begin
        logic [LANES-1:0]       v, tv;
        logic [LANES-1:0][31:0] d, td;
        v = w_req[p]; d = w_data[p]; tv = '0; td = '0;
        for (int k = 0; k < BW; k++)
          if (rot[k]) begin
            tv = v; td = d;
            for (int l = 0; l < LANES; l++) begin
              v[(l + (1 << k)) % LANES] = tv[l];
              d[(l + (1 << k)) % LANES] = td[l];
            end
          end
        pv[p] = v; pdt[p] = d;
        for (int b = 0; b < LANES; b++) pa[p][b] = a0[BW +: IW] + IW'(BW'(b) < rot);
      end
    end
  end
  always_comb begin
    bw_v = '0; bw_a = '0; bw_d = '0;
    for (int p = 0; p < NWP; p++)
      if (w_gnt[p])
        for (int b = 0; b < LANES; b++)
          if (pv[p][b]) begin
            bw_v[b] = 1'b1;
            bw_a[b] = bw_a[b] | pa[p][b];
            bw_d[b] = bw_d[b] | pdt[p][b];
          end
  end

  // ---- the registered write stage (otpu_tmem): lands at the next cycle's odd fast edge here.
  // FW: its registers are on clk_f, loaded at the even fast edges only (the core edges, where
  // otpu_tmem loads them): the same values at the same instants, but their inputs then cross from
  // the core domain in two fast periods (like the read addresses) and only one-level fast-domain
  // paths reach the RAMs. The even edges come from a phase of its own (pw_ev, as the banks').
  (* max_fanout = 16 *) logic [LANES-1:0]         pw_v;
  (* max_fanout = 16 *) logic [LANES-1:0][IW-1:0] pw_a;
  (* max_fanout = 16 *) logic [LANES-1:0][31:0]   pw_d;
  logic [LANES-1:0][31:0]                         pd;
  if (FW) begin : g_fw
    (* keep *) logic wtog;
    (* keep *) logic wtq;
    (* max_fanout = 32 *) logic pw_ev;
    always_ff @(posedge clk) wtog <= !wtog;
    always_ff @(posedge clk_f) begin
      wtq <= wtog;
      pw_ev <= wtq != wtog;          // the next fast edge is even
    end
    always_ff @(posedge clk_f)
      if (pw_ev) begin
        pw_v <= bw_v;
        pw_a <= bw_a;
        pw_d <= bw_d;
      end
`ifndef SYNTHESIS
    initial begin wtog = 1'b0; wtq = 1'b0; pw_ev = 1'b1; end
`endif
  end else begin : g_sw
    always_ff @(posedge clk) begin
      pw_v <= bw_v;
      pw_a <= bw_a;
      pw_d <= bw_d;
    end
  end
  always_ff @(posedge clk)
    if (BYP) pd <= pw_d;
`ifndef SYNTHESIS
  initial pw_v = '0;
`endif

  // ---- read ports' bank enables and rows (otpu_tmem)
  logic [NRP-1:0][LANES-1:0]         pb_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] pb_a;
  for (genvar p = 0; p < NRP; p++) begin : g_preq
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    wire  [BW+IW-1:0]         a0 = r_addr[p][0][BW+IW-1:0];
    wire  [BW-1:0]            rot = a0[BW-1:0];
    wire  [LANES-1:0]         rs = (IMPL == 1 && !RQ) ? r_en[p] : r_req[p];
    always_comb begin
      b_en = '0; b_a = '0;
      if (GEN_R[p]) begin
        for (int b = 0; b < LANES; b++)
          for (int l = 0; l < LANES; l++)
            if (r_addr[p][l][BW-1:0] == BW'(b) && rs[l]) begin
              b_en[b] = 1'b1;
              b_a[b] = b_a[b] | r_addr[p][l][BW +: IW];
            end
      end else begin
        logic [LANES-1:0] t;
        t = '0;
        b_en = rs;
        for (int k = 0; k < BW; k++)
          if (rot[k]) begin
            t = b_en;
            for (int l = 0; l < LANES; l++) b_en[(l + (1 << k)) % LANES] = t[l];
          end
        for (int b = 0; b < LANES; b++) b_a[b] = a0[BW +: IW] + IW'(BW'(b) < rot);
      end
    end
    assign pb_en[p] = b_en;
    assign pb_a[p] = b_a;
  end

  // ---- each host's bank enables and rows (otpu_tmem's g_copy: IMPL 1 the granted port on its
  // slot, IMPL 0 the host or, at SH_HOST, the first guest that asks)
  logic [NRP-1:0][LANES-1:0]         h_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] h_a;
  for (genvar p = 0; p < NRP; p++) begin : g_host
    if (host_of(p) == p) begin : g_h
      if (IMPL == 1 && RQ) begin : g_bq
        always_comb begin
          h_en[p] = '0; h_a[p] = '0;
          for (int b = 0; b < LANES; b++)
            for (int r = NRP - 1; r >= 0; r--)
              for (int g = 0; g < NRP; g++)
                if (CPY[g] == CPY[p] && int'(PRANK[g]) == r && pb_en[g][b]) begin
                  h_en[p][b] = 1'b1;
                  h_a[p][b] = pb_a[g][b];
                end
        end
      end else if (IMPL == 1) begin : g_bk
        always_comb begin
          h_en[p] = '0; h_a[p] = '0;
          for (int g = 0; g < NRP; g++)
            if (CPY[g] == CPY[p])
              for (int b = 0; b < LANES; b++)
                if (pb_en[g][b]) begin
                  h_en[p][b] = 1'b1;
                  h_a[p][b] = h_a[p][b] | pb_a[g][b];
                end
        end
      end else if (p == SH_HOST && SH_MASK != '0) begin : g_sh
        always_comb begin
          logic take;
          h_en[p] = pb_en[p]; h_a[p] = pb_a[p];
          take = |r_req[p];
          for (int g = 0; g < NRP; g++)
            if (SH_MASK[g] && !take && |r_req[g]) begin
              h_en[p] = pb_en[g]; h_a[p] = pb_a[g]; take = 1'b1;
            end
        end
      end else begin : g_own
        assign h_en[p] = pb_en[p];
        assign h_a[p] = pb_a[p];
      end
    end else begin : g_g
      assign h_en[p] = '0;
      assign h_a[p] = '0;
    end
  end

  // ---- the pumped copies: copy c's bank b serves slot 2c on port A and slot 2c + 1 on port B
  logic [NRP-1:0][LANES-1:0][31:0] hq;     // each host's bank data
  for (genvar c = 0; c < NPC; c++) begin : g_pc
    localparam int HA = host_on(2 * c);
    localparam int HB = host_on(2 * c + 1);
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [31:0] qa, qb;
      otpu_pump_bank #(.N(BD), .STYLE(STYLE), .FW(FW)) u_pb (
        .clk, .clk_f, .wv(pw_v[b]), .wa(pw_a[b]), .wd(pw_d[b]),
        .rena(HA >= 0 ? h_en[HA < 0 ? 0 : HA][b] : 1'b0), .ra(h_a[HA < 0 ? 0 : HA][b]),
        .renb(HB >= 0 ? h_en[HB < 0 ? 0 : HB][b] : 1'b0), .rb(h_a[HB < 0 ? 0 : HB][b]),
        .qa, .qb);
      if (HA >= 0) begin : g_a
        assign hq[HA][b] = qa;
      end
      if (HB >= 0) begin : g_b
        assign hq[HB][b] = qb;
      end
    end
  end
  for (genvar p = 0; p < NRP; p++) begin : g_nohost
    if (host_of(p) != p) begin : g_z
      assign hq[p] = '0;
    end
  end

  // ---- per port: the bypass and the lanes' data path (otpu_tmem)
  logic [NRP-1:0][LANES-1:0][31:0] pq;
  for (genvar p = 0; p < NRP; p++) begin : g_port
    if (host_of(p) == p) begin : g_copy
      logic [LANES-1:0][31:0] qb;
      for (genvar b = 0; b < LANES; b++) begin : g_bank
        if (BYP) begin : g_byp
          logic bh;
          always_ff @(posedge clk)
            if (h_en[p][b]) bh <= pw_v[b] && pw_a[b] == h_a[p][b];
          assign qb[b] = bh ? pd[b] : hq[p][b];
`ifndef SYNTHESIS
          initial bh = 1'b0;
`endif
        end else begin : g_nob
          assign qb[b] = hq[p][b];
        end
      end
      assign pq[p] = qb;
    end else begin : g_guest
      assign pq[p] = pq[host_of(p)];
    end
    wire  [BW-1:0]            rot = r_addr[p][0][BW-1:0];
    wire  [LANES-1:0][31:0]   qb = pq[p];
    logic [LANES-1:0][BW-1:0] sel;
    logic [LANES-1:0][BW-1:0] rq;
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held, lq;
    if (GEN_R[p]) begin : g_xb
      assign rq = '0;
    end else begin : g_rot
      (* keep *) logic [LANES-1:0][BW-1:0] rot_q;
      always_ff @(posedge clk)
        for (int l = 0; l < LANES; l++) if (|r_en[p]) rot_q[l] <= rot;
      assign rq = rot_q;
    end
    always_ff @(posedge clk)
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= r_en[p][l];
        if (GEN_R[p] && r_en[p][l]) sel[l] <= r_addr[p][l][BW-1:0];
        if (fresh[l]) held[l] <= lq[l];
      end
    always_comb
      for (int l = 0; l < LANES; l++) begin
        lq[l] = GEN_R[p] ? qb[sel[l]] : qb[(l + int'(rq[l])) % LANES];
        r_data[p][l] = fresh[l] ? lq[l] : held[l];
      end
  end

`ifndef SYNTHESIS
  // IMPL 1: at most one granted port per slot and bank (the slice's arbiter, gnt_k; RQ: the
  // requests may meet, the arbiter grants the first)
  always @(posedge clk)
    if (IMPL == 1 && !RQ)
      for (int b = 0; b < LANES; b++)
        for (int p = 0; p < NRP; p++)
          for (int q = p + 1; q < NRP; q++)
            if (CPY[p] == CPY[q] && pb_en[p][b] && pb_en[q][b])
              $fatal(1, "otpu_tmem_pump: ports %0d and %0d granted in slot %0d bank %0d", p, q,
                     CPY[p], b);
`endif
endmodule

// One bank of a pumped copy: a true dual-port RAM on clk_f (= 2 x clk, aligned). Even fast edges
// (the core edges): port A reads ra (rena), port B reads rb (renb). Odd fast edges (mid core
// cycle): port A writes wd at wa (wv), its output held (NO_CHANGE). qa / qb change only after even
// edges and hold through the next core cycle. Its hierarchy is kept: the timing contract names its
// read inputs (ooc.tcl).
(* keep_hierarchy = "yes" *)
module otpu_pump_bank #(
  parameter int    N     = 8192,
  parameter string STYLE = "block",
  parameter bit    FW    = 1'b1      // the phase from a fast-domain register (below)
) (
  input  logic                 clk,
  input  logic                 clk_f,
  input  logic                 wv,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [31:0]          wd,
  input  logic                 rena,
  input  logic [$clog2(N)-1:0] ra,
  input  logic                 renb,
  input  logic [$clog2(N)-1:0] rb,
  output logic [31:0]          qa,
  output logic [31:0]          qb
);
  // the phase: tog flips at every core edge; tq is its copy a fast edge later. They differ during
  // a core cycle's first half (the next fast edge is odd) and agree in its second (even next).
  // FW: ev is a fast-domain register loaded with that at every fast edge (the edge after next's
  // parity is the opposite of the next's), so no core-domain net reaches the RAM's enables; it
  // aligns itself at the first fast edge, as tq does.
  (* keep *) logic tog;
  (* keep *) logic tq;
  logic evr;
  always_ff @(posedge clk) tog <= !tog;
  always_ff @(posedge clk_f) begin
    tq <= tog;
    evr <= tq != tog;
  end
  wire ev = FW ? evr : (tq == tog);
`ifndef SYNTHESIS
  initial begin tog = 1'b0; tq = 1'b0; evr = 1'b1; end
`endif
  wire                 en_a = ev ? rena : wv;
  wire                 we_a = !ev && wv;
  wire [$clog2(N)-1:0] ad_a = ev ? ra : wa;
  wire                 en_b = ev && renb;
  if (STYLE == "ultra") begin : g_u
    (* ram_style = "ultra" *) logic [31:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) qb <= mem[rb];
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_b
    (* ram_style = "block" *) logic [31:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) qb <= mem[rb];
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule
