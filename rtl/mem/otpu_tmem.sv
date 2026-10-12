// TMEM: fp32 scratchpad organised as LANES banks (bank = word address mod LANES). Units present
// lane-wide requests on their own ports; the slice's arbiter grants each unit all-or-nothing
// per cycle so that no bank takes more than WPB writes (and RPB reads) per cycle, and a unit
// that is not granted holds its state.
//
// Implementation (the same for simulation and the FPGA): every read port has its own copy of
// the memory, LANES banks of WORDS/LANES words, so reads never compete across ports -- except
// the guest ports (SH_MASK), rarely busy, which share one port's copy in priority order. Writes are
// broadcast to every copy: each bank takes up to WPB writes per cycle, picked from all write
// ports' lanes (WPB = 1 on the FPGA, where each bank copy is a simple dual-port BRAM with
// read-first behaviour). A read returns the old word if it is written in the same cycle.
// Each read lane's data is held until that lane reads again, so a frozen unit keeps its
// operands.
//
// Timing: the grants are the end of the slice's deepest logic (unit requests -> arbiter), and
// a write reaches every copy of its bank (NRP * the bank's block RAMs, spread over the die). So
// the selected writes are registered here and land in the block RAMs a cycle later, and a read
// of a word whose write is still in that register takes the registered data (a bypass): to the
// units the memory behaves exactly as if the write had landed in its own cycle. Likewise the
// block RAM read enables come from the requests, not the grants: a read that is not granted
// only changes a block RAM output nobody looks at (the lanes take it only the cycle after a
// granted read, and hold it otherwise).
//
// Lane routing. Every port except those in GEN_R / GEN_W (the MXU's: its drain writes scattered
// rows) carries a contiguous run: enabled lane l holds word a0 + l, where a0 is lane 0's
// address (the units drive it whenever any lane is enabled). Such a port reaches the banks
// through barrel rotators by a0 mod LANES (the lanes' bank rows are a0's row, +1 for the banks before a0's), LANES x
// log2(LANES) multiplexers instead of a LANES x LANES crossbar; the simulation stops if a
// rotator port's lanes are not contiguous. The GEN ports keep the crossbar.
//
// Within one port the enabled lanes must hit distinct banks (or read the same word); the
// simulation stops on a violation, so the timing it reports is honest.
//
// IMPL 1 (banked; an ASIC, whose SRAM macros cannot be replicated per read port at no cost):
// NCP copies instead of one per read port, each LANES banks of one 1R1W RAM (otpu_ram_1r1w, the
// target's technology cell). CPY[p] is read port p's copy; ports of one unit (the slice's
// arbitration group) sit on different copies, so a unit alone always fits. Each copy's bank
// serves one read per cycle: the slice's arbiter admits at most one port per copy and bank (a
// port's lanes in one bank read one word), and the bank's address comes from that port's granted
// lanes (r_en), not its requests. Writes are broadcast to every copy, one per bank (WPB = 1).
// The registered write stage, the bypass and the lanes' data path are IMPL 0's.
// RQ (IMPL 1): a copy's bank address comes from the requests instead, the port first in the
// slice's priority order (PRANK, smaller first) among those on the copy that ask for the bank;
// the slice's arbiter then grants a unit only where it is that port (otpu_slice TMEM_RQ), so the
// granted port's address is the one at the RAM and the grants reach no RAM pin (on the FPGA, the
// grant-to-address path of IMPL 1 does not close at the core clock: docs/tmem_pump.md).
module otpu_tmem #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 6,
  parameter int NWP   = 4,
  parameter int WPB   = 1,
  parameter logic [NRP-1:0] GEN_R = '1,   // read ports with the general crossbar
  parameter logic [NWP-1:0] GEN_W = '1,   // write ports with the general crossbar (WPB = 1)
  parameter int SH_HOST = 0,              // the read port whose copy SH_MASK's ports share
  parameter logic [NRP-1:0] SH_MASK = '0, // guests on SH_HOST's copy (the arbiter lets a guest
                                          // read only when no port before it on the copy asks)
  parameter int SID   = 0,
  parameter int IMPL  = 0,                // 0: a copy per read port (block RAM); 1: banked (see top)
  parameter int NCP   = 1,                // IMPL 1: copies
  parameter logic [NRP-1:0][2:0] CPY = '0, // IMPL 1: each read port's copy (< NCP)
  parameter bit   RQ    = 1'b0,           // IMPL 1: bank addresses from the requests (see top)
  parameter logic [NRP-1:0][2:0] PRANK = '0, // RQ, IMPL 3: each read port's priority (0 first)
  parameter int   NSB   = 8,              // IMPL 3: RAMs (sub-banks) per bank
  parameter int   SBP   = 3,              // IMPL 3: the sub-bank field's lowest word-address bit
  parameter bit   SHARE = 1'b1,           // IMPL 3: ports reading one word share its RAM read
  parameter logic [NRP-1:0][2:0] GRP = '0, // IMPL 3: each read port's group (the slice's unit)
  parameter int   SBH   = 0,              // IMPL 3: the sub-bank hashed with the row bits above
                                          //   it (1: the next field, 2: every field, XORed)
  parameter bit   RPL   = 1'b0,           // IMPL 3: a lost read is replayed the next cycle (see
                                          //   IMPL 3 below)
  parameter logic [NRP-1:0] RPLM = '1,    // RPL: the ports whose reads may be replayed
  parameter int   TOPP  = 0,              // RPL: the port whose request comes before the replays
  parameter bit   TDP   = 1'b0,           // IMPL 3: true dual-port RAMs (see IMPL 3 below)
  parameter logic [NRP-1:0] SIDE = '0     // TDP: each read port's RAM port (1: A, the writes')
) (
  input  logic                                clk,
  input  logic [NRP-1:0][LANES-1:0]           r_en,     // granted reads
  input  logic [NRP-1:0][LANES-1:0]           r_req,    // requested reads (r_en before the grant)
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,     // granted writes
  input  logic [NWP-1:0][LANES-1:0]           w_req,    // requested writes (w_en before the grant)
  input  logic [NWP-1:0]                      w_gnt,    // the grant of each write port's unit
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_addr,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_data,
  output logic [NRP-1:0]                      blk_x,    // IMPL 3: a port of another group
  output logic [NRP-1:0]                      blk_s,    //   / of its own took a RAM it reads
  output logic [NRP-1:0]                      lost,     // IMPL 3 RPL: the port's read this cycle
                                                        //   lost a RAM (replayed next cycle)
  output logic [NRP-1:0]                      frz,      // IMPL 3 RPL: the port replays now
  input  logic                                dump
);
  localparam int BW = $clog2(LANES);
  localparam int BD = WORDS / LANES;        // words per bank
  localparam int IW = $clog2(BD);
  initial if ((IMPL == 1 || IMPL == 3) && WPB != 1) $fatal(1, "otpu_tmem: IMPL %0d needs WPB = 1", IMPL);
  initial if (TDP && IMPL != 3) $fatal(1, "otpu_tmem: TDP needs IMPL 3");
  initial if (IMPL == 3 && (NSB < 2 || (1 << $clog2(NSB)) != NSB || SBP < BW || SBP + $clog2(NSB) > BW + IW))
    $fatal(1, "otpu_tmem: IMPL 3 sub-banks NSB %0d (a power of 2, >= 2) at bit SBP %0d", NSB, SBP);
  initial if (IMPL == 1) for (int p = 0; p < NRP; p++)
    if (CPY[p] >= NCP) $fatal(1, "otpu_tmem: port %0d's copy %0d >= NCP %0d", p, CPY[p], NCP);
  // the read port that holds each port's copy: IMPL 0, the port itself or SH_HOST (a guest);
  // IMPL 1, the first port on its copy
  function automatic int host_of(input int p);
    if (IMPL == 0) return SH_MASK[p] ? SH_HOST : p;
    for (int q = 0; q < NRP; q++) if (CPY[q] == CPY[p]) return q;
    return p;
  endfunction


  // ---- write selection: per bank, the enabled lanes of all write ports. The arbiter admits at
  // most WPB of them, so with WPB = 1 a flat AND-OR tree selects the one (no priority chain).
  logic [LANES-1:0][WPB-1:0]         bw_v;
  logic [LANES-1:0][WPB-1:0][IW-1:0] bw_a;
  logic [LANES-1:0][WPB-1:0][31:0]   bw_d;
  if (WPB == 1) begin : g_w1
    // each port's lanes are merged per bank from its requests (a port belongs to one unit, its
    // lanes hit distinct banks), then the ports by their grants: the grant enters at the last
    // AND-OR level instead of at every lane (the arbiter admits one granted writer per bank)
    // per port, its lanes in bank order (a rotator, or the crossbar of the port's lanes, which
    // hit distinct banks), then the ports merged by their grants: the grant enters at the last
    // AND-OR level instead of at every lane (the arbiter admits one granted writer per bank)
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
              bw_v[b][0] = 1'b1;
              bw_a[b][0] = bw_a[b][0] | pa[p][b];
              bw_d[b][0] = bw_d[b][0] | pdt[p][b];
            end
    end
  end else begin : g_wn
    always_comb begin
      for (int b = 0; b < LANES; b++) begin
        int n;
        n = 0;
        bw_v[b] = '0; bw_a[b] = '0; bw_d[b] = '0;
        for (int p = 0; p < NWP; p++)
          for (int l = 0; l < LANES; l++)
            if (w_en[p][l] && w_addr[p][l][BW-1:0] == BW'(b) && n < WPB) begin
              bw_v[b][n] = 1'b1;
              bw_a[b][n] = w_addr[p][l][BW +: IW];
              bw_d[b][n] = w_data[p][l];
              n = n + 1;
            end
      end
    end
  end

  // ---- the registered write stage: this cycle's selected writes land next cycle (pw_*); the
  // data of the writes landing now is kept one more cycle (pd) for the reads that bypass them.
  // Each pw bit reaches every copy of its bank (the copies' block RAMs spread over the die):
  // synthesis replicates the registers so each copy gets nearby ones (one copy's net to the
  // block RAM address pins was the worst core_clk path at 120 MHz, 0 logic levels)
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0]         pw_v;
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0][IW-1:0] pw_a;
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0][31:0]   pw_d;
  logic [LANES-1:0][WPB-1:0][31:0]                         pd;
  always_ff @(posedge clk) begin
    pw_v <= bw_v;
    pw_a <= bw_a;
    pw_d <= bw_d;
    pd <= pw_d;
  end
`ifndef SYNTHESIS
  initial pw_v = '0;
`endif

  // ---- read ports: each port's bank enables and rows (from its requests)
  logic [NRP-1:0][LANES-1:0]         pb_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] pb_a;
  logic [NRP-1:0][LANES-1:0][31:0]   pq;      // the bank data each port's copy returns
  logic [NRP-1:0][LANES-1:0]         rpf;     // IMPL 3 RPL: the lanes a replay reads now
  for (genvar p = 0; p < NRP; p++) begin : g_preq
    // per bank: the address of the lane that reads it (lanes of one port hit distinct banks or
    // read the same word, so an AND-OR merge is exact). A port belongs to one unit and the
    // grant is all-or-nothing per unit, so the address and the enable are merged from the
    // requests: the grant reaches no block RAM pin (it only decides which lanes take the data)
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    wire  [BW+IW-1:0]         a0 = r_addr[p][0][BW+IW-1:0];
    wire  [BW-1:0]            rot = a0[BW-1:0];
    // IMPL 1: a copy's bank serves the one port granted in it, so its enables are the grants
    // (RQ: the requests, the copy picks by priority)
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

`ifndef SYNTHESIS
  // ---- the dump (simulation): the memory itself, not the writes asked of it. On `dump` the first
  // copy's banks (DP) put their words into img, word i = bank i mod LANES's row i / LANES; a cycle
  // later every other copy is checked against them (a copy that differs stops the simulation)
  // and img goes to <dir>/tmem_<SID>.hex. +tmem_fault=<SID>,<word>: every write to the word
  // lands with bit 0 flipped, in every copy, or with +tmem_fault_port=<port> in that port's
  // (tests: the dump and the check see a fault in the registered write stage, the RAM or one
  // copy)
  function automatic int first_copy();
    for (int p = 0; p < NRP; p++) if (host_of(p) == p) return p;
    return 0;
  endfunction
  localparam int DP = first_copy();
  logic [31:0] img [WORDS];
  logic        dump_q;
  int          fault_w, fault_p;
  initial begin
    string arg;
    int sid, w, p;
    dump_q = 1'b0; fault_w = -1; fault_p = -1;
    if ($value$plusargs("tmem_fault=%s", arg) && $sscanf(arg, "%d,%d", sid, w) == 2 && sid == SID)
    begin
      fault_w = w;
      if ($value$plusargs("tmem_fault_port=%d", p)) fault_p = p;
    end
  end
  always @(posedge clk) dump_q <= dump;
  function automatic logic [31:0] fault(input int p, input int b, input logic [IW-1:0] row);
    return 32'(fault_w >= 0 && fault_w == int'(row) * LANES + b && (fault_p < 0 || fault_p == p));
  endfunction
`endif

  // ---- the copies: one per read port, except the guests (SH_MASK), which read SH_HOST's copy:
  // it serves the first port that requests, the host, then the guests in port order. IMPL 1:
  // one per CPY value, held by its first port; each bank serves the port granted in it.
  // ---- IMPL 3 (EXPERIMENTAL, docs/tmem_nocopy.md): one copy. Each bank is NSB RAMs with their
  // own addresses: a word's sub-bank is its bank row's bits [SBP-BW +: log2 NSB] (SBP = BW: the
  // 8-word groups interleave over the RAMs; higher: regions), XORed with the row's fields above
  // them (SBH 1: the next one, 2: all of them: a hash). A RAM's address is that of the asker
  // first in rank, from the requests (no grant on a RAM's address path, as RQ). An asker that
  // asks a RAM an asker before it took for another word (SHARE: one word read by two is one read)
  // loses it.
  // Without RPL the ranks are PRANK's and a port that would lose is blocked: by a port of
  // another group (blk_x: the slice does not grant the unit) or of its own (blk_s: the slice reads
  // the group's ports over successive cycles, otpu_slice TMEM_IMPL 3).
  // RPL (replay): the units read as if no RAM were shared (the slice's grant ignores the RAMs) and
  // a read that lost is read again the next cycle: the port's banks and rows are registered with
  // the read, and the next cycle they ask again (frz) ranked before every request but TOPP's (the
  // DMA's, which has no hold); their words reach the port's lanes the cycle after (the lanes
  // fresh again, without the bypass: the replay returns what the first read would have, the
  // writes landed since are the ones that were before it). `lost` tells the slice to hold the
  // port's unit for the cycle the replay takes (its lanes' words of the lost read are not the
  // ones asked). blk_x is then a port's request losing (for units that cannot hold, which the
  // slice keeps from reading instead), blk_s 0.
  // A port asks with its request, or with its replay while one is pending (the slice then holds
  // its unit, so its request does not ask): one asker a port, whose rank the replay registers pick.
  // Writes: one per bank and cycle, as IMPL 0 and 1, to the RAM of its sub-bank; the registered
  // write stage and its bypass are IMPL 0's.
  // TDP: true dual-port RAMs, each read port on one of a RAM's two ports (SIDE: 0 port B, 1 port
  // A), ranked among the ports on its side only; a write takes its RAM's port A, so an asker on
  // port A of a RAM being written loses unless it reads the written word (a read takes the
  // bypass, a replay port A's read-first word). On the 7-series a true dual-port RAMB36 is at most
  // 36 bits wide: a RAM of 1K words is one RAMB36, as in the simple dual-port.
  // an asker's rank (0 first): without RPL PRANK's; with it TOPP's request, the replays in PRANK
  // order, then the other requests in PRANK order. rlive: the port's reads may be replayed
  function automatic int rnk(input int p, input bit rp);
    if (!RPL) return int'(PRANK[p]);
    if (rp) return 1 + int'(PRANK[p]);
    if (p == TOPP) return 0;
    return 1 + NRP + int'(PRANK[p]);
  endfunction
  function automatic bit rlive(input int p);
    return RPL && RPLM[p] && p != TOPP;
  endfunction
  function automatic int sd_of(input int p);   // the RAM port a read port asks on (0: B)
    return (TDP && SIDE[p]) ? 1 : 0;
  endfunction
  // OTPU_TMEM_ABL_* (EXPERIMENTAL, out-of-context LUT breakdown only, docs/tmem_nocopy.md 6.3: the
  // TMEM is wrong with any of them): _DSEL a port takes sub-bank 0's word (no data select), _CMP no
  // conflict compares (nothing lost), _RPL no replay buffers (an asker is its request), _AMUX a
  // RAM's address is its side's first port's (no priority multiplexer)
  if (IMPL == 3) begin : g_one
    localparam int LSB = $clog2(NSB);
    localparam int SO  = SBP - BW;             // the sub-bank field's offset in a bank row
    localparam int RW  = IW - LSB;             // a RAM's address bits
    localparam int RD  = BD / NSB;             // a RAM's words
    localparam int NS  = TDP ? 2 : 1;          // a RAM's read ports
    function automatic logic [LSB-1:0] hash(input logic [IW-1:0] hi);
      logic [LSB-1:0] x;
      x = '0;
      if (SBH == 1) x = LSB'(hi);
      if (SBH == 2) for (int f = 0; f < IW; f += LSB) x = x ^ LSB'(hi >> f);
      return x;
    endfunction
    function automatic logic [RW-1:0] ram_row(input logic [IW-1:0] row);
      logic [IW-1:0] hi, lo;
      hi = row >> (SO + LSB);
      lo = row & ((IW'(1) << SO) - IW'(1));
      return RW'((hi << SO) | lo);
    endfunction
    function automatic logic [LSB-1:0] sub_of(input logic [IW-1:0] row);
      return row[SO +: LSB] ^ hash(row >> (SO + LSB));
    endfunction
`ifndef SYNTHESIS
    // the bank row held in RAM k's row r (the dump)
    function automatic int row_of(input int k, input int r);
      logic [IW-1:0] hi, lo, f;
      hi = IW'(r >> SO);
      lo = IW'(r & ((1 << SO) - 1));
      f = IW'(LSB'(k) ^ hash(hi));
      return int'((hi << (SO + LSB)) | (f << SO) | lo);
    endfunction
`endif
    // the replays: a lost read's banks, rows and lanes, registered with it
    logic [NRP-1:0]                    rpv;
    logic [NRP-1:0][LANES-1:0]         rpe;
    logic [NRP-1:0][LANES-1:0][IW-1:0] rpa;
    logic [NRP-1:0][LANES-1:0]         rpl;
    logic [NRP-1:0]                    rp;      // the port's asker is its replay
    // each asker's banks, bank rows, sub-banks and RAM rows
    logic [NRP-1:0][LANES-1:0]          ae;
    logic [NRP-1:0][LANES-1:0][IW-1:0]  aa;
    logic [NRP-1:0][LANES-1:0][LSB-1:0] asb;
    logic [NRP-1:0][LANES-1:0][RW-1:0]  arw;
    for (genvar i = 0; i < NRP; i++) begin : g_as
`ifdef OTPU_TMEM_ABL_RPL
      assign rp[i] = 1'b0;
`else
      assign rp[i] = rlive(i) && rpv[i];
`endif
      for (genvar b = 0; b < LANES; b++) begin : g_b
        assign ae[i][b] = rp[i] ? rpe[i][b] : pb_en[i][b];
        assign aa[i][b] = rp[i] ? rpa[i][b] : pb_a[i][b];
        assign asb[i][b] = sub_of(aa[i][b]);
        assign arw[i][b] = ram_row(aa[i][b]);
      end
    end
    // bf[i][o]: asker o ranks before asker i this cycle (on i's side; the replays' registers pick
    // the case of four constant ones)
    logic [NRP-1:0][NRP-1:0] bf;
    for (genvar i = 0; i < NRP; i++) begin : g_bf
      for (genvar o = 0; o < NRP; o++) begin : g_o
        if (o == i || sd_of(o) != sd_of(i)) begin : g_n
          assign bf[i][o] = 1'b0;
        end else begin : g_y
          localparam bit B00 = rnk(o, 1'b0) < rnk(i, 1'b0);
          localparam bit B01 = rnk(o, 1'b0) < rnk(i, 1'b1);
          localparam bit B10 = rnk(o, 1'b1) < rnk(i, 1'b0);
          localparam bit B11 = rnk(o, 1'b1) < rnk(i, 1'b1);
          assign bf[i][o] = rp[o] ? (rp[i] ? B11 : B10) : (rp[i] ? B01 : B00);
        end
      end
    end
    // wk[i][b]: TDP, asker i asks port A of a RAM the write in bank b takes
    logic [NRP-1:0][LANES-1:0] wk;
    for (genvar i = 0; i < NRP; i++) begin : g_wk
      for (genvar b = 0; b < LANES; b++) begin : g_b
        if (sd_of(i) == 1) begin : g_a
          assign wk[i][b] = pw_v[b][0] && sub_of(pw_a[b][0]) == asb[i][b];
        end else begin : g_b0
          assign wk[i][b] = 1'b0;
        end
      end
    end
    // the RAMs' addresses: per RAM and side, the asker first in rank (an AND-OR multiplexer of
    // the askers' rows)
    logic [LANES-1:0][NSB-1:0][NS-1:0]         s_en;
    logic [LANES-1:0][NSB-1:0][NS-1:0][RW-1:0] s_a;
    for (genvar b = 0; b < LANES; b++) begin : g_ra
      for (genvar k = 0; k < NSB; k++) begin : g_k
        logic [NRP-1:0] rq, win;
        for (genvar i = 0; i < NRP; i++) begin : g_i
          assign rq[i] = ae[i][b] && asb[i][b] == LSB'(k);
          assign win[i] = rq[i] && !(|(rq & bf[i]));
        end
        always_comb
          for (int s = 0; s < NS; s++) begin
            bit got;
            got = 1'b0;
            s_en[b][k][s] = 1'b0;
            s_a[b][k][s] = '0;
            for (int i = 0; i < NRP; i++)
              if (sd_of(i) == s) begin
                s_en[b][k][s] = s_en[b][k][s] | rq[i];
`ifdef OTPU_TMEM_ABL_AMUX
                if (!got) s_a[b][k][s] = arw[i][b];
                got = 1'b1;
`else
                if (win[i]) s_a[b][k][s] = s_a[b][k][s] | arw[i][b];
`endif
              end
          end
      end
    end
    // cp[i][o] (o < i, one side): the two askers ask one RAM for different words; hm[i][o]: o,
    // ranked before i, does so (on port A of a RAM being written the write decides: wm)
    logic [NRP-1:0][NRP-1:0] cp, hm;
    logic [NRP-1:0]          wm;
    for (genvar i = 0; i < NRP; i++) begin : g_cf
      for (genvar o = 0; o < NRP; o++) begin : g_o
        if (o < i && sd_of(o) == sd_of(i)) begin : g_c
          logic [LANES-1:0] hb;
          for (genvar b = 0; b < LANES; b++) begin : g_b
            assign hb[b] = ae[i][b] && ae[o][b] && asb[o][b] == asb[i][b] && !wk[i][b] &&
                           (!SHARE || aa[o][b] != aa[i][b]);
          end
`ifdef OTPU_TMEM_ABL_CMP
          assign cp[i][o] = 1'b0;
`else
          assign cp[i][o] = |hb;
`endif
        end else begin : g_n
          assign cp[i][o] = 1'b0;
        end
      end
      logic [LANES-1:0] wb;
      for (genvar b = 0; b < LANES; b++) begin : g_w
        assign wb[b] = ae[i][b] && wk[i][b] && pw_a[b][0] != aa[i][b];
      end
`ifdef OTPU_TMEM_ABL_CMP
      assign wm[i] = 1'b0;
`else
      assign wm[i] = |wb;
`endif
    end
    always_comb
      for (int i = 0; i < NRP; i++)
        for (int o = 0; o < NRP; o++)
          hm[i][o] = (o < i ? cp[i][o] : cp[o][i]) && bf[i][o];
    always_comb
      for (int p = 0; p < NRP; p++) begin
        blk_x[p] = wm[p]; blk_s[p] = 1'b0;
        for (int o = 0; o < NRP; o++)
          if (hm[p][o]) begin
            if (!RPL && GRP[o] == GRP[p]) blk_s[p] = 1'b1;
            else blk_x[p] = 1'b1;
          end
      end
    // a read lost: the port read (its unit was granted, r_en) or replayed, and lost a RAM
    always_comb
      for (int p = 0; p < NRP; p++)
        lost[p] = rlive(p) && (rpv[p] || (|r_en[p])) && ((|hm[p]) || wm[p]);
    always_ff @(posedge clk)
      for (int p = 0; p < NRP; p++) begin
        rpv[p] <= lost[p];
        if (!rpv[p]) begin rpe[p] <= pb_en[p]; rpa[p] <= pb_a[p]; rpl[p] <= r_en[p]; end
      end
`ifndef SYNTHESIS
    initial rpv = '0;
    // the replays' premise: no write lands on a word awaiting its replay before the replay wins
    // (it would read the new word). A write landing in the cycle the replay wins is fine (the
    // RAM reads first); one landing while it loses again is not.
    always @(posedge clk)
      for (int p = 0; p < NRP; p++)
        if (RPL && rpv[p] && lost[p])
          for (int b = 0; b < LANES; b++)
            if (rpe[p][b] && pw_v[b][0] && pw_a[b][0] == rpa[p][b])
              $fatal(1, "TMEM%0d: port %0d's word %0d awaiting a replay was written at %0t", SID, p,
                     int'(rpa[p][b]) * LANES + b, $time);
`endif
    assign frz = RPL ? rpv : '0;
    always_comb
      for (int p = 0; p < NRP; p++) rpf[p] = (RPL && rpv[p]) ? rpl[p] : '0;
    logic [LANES-1:0][NSB-1:0][NS-1:0][31:0] s_q;
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      for (genvar k = 0; k < NSB; k++) begin : g_sub
        logic [31:0] wd;
`ifdef SYNTHESIS
        assign wd = pw_d[b][0];
`else
        assign wd = pw_d[b][0] ^ fault(0, b, pw_a[b][0]);
`endif
        wire wh = pw_v[b][0] && sub_of(pw_a[b][0]) == LSB'(k);
        // (inferred here: 64 otpu_ram_1r1w cells of this size in the slice crash Verilator 5.046's
        // front end, an internal fault; the cell is this RAM)
        (* ram_style = "block" *) logic [31:0] mem [RD];
        if (TDP) begin : g_tdp
          // port A: the write, else its side's read; read-first (a replay of the written word
          // takes the word before the write)
          wire [RW-1:0] ra = wh ? ram_row(pw_a[b][0]) : s_a[b][k][1];
          always_ff @(posedge clk)
            if (wh || s_en[b][k][1]) begin
              if (wh) mem[ra] <= wd;
              s_q[b][k][1] <= mem[ra];
            end
          always_ff @(posedge clk)
            if (s_en[b][k][0]) s_q[b][k][0] <= mem[s_a[b][k][0]];
        end else begin : g_sdp
          always_ff @(posedge clk) begin
            if (wh) mem[ram_row(pw_a[b][0])] <= wd;
            if (s_en[b][k][0]) s_q[b][k][0] <= mem[s_a[b][k][0]];
          end
        end
`ifndef SYNTHESIS
        initial for (int r = 0; r < RD; r++) mem[r] = '0;
        always @(posedge clk)
          if (dump)
            for (int r = 0; r < RD; r++) img[row_of(k, r) * LANES + b] = mem[r];
`endif
      end
    end
    // each port's data: per bank, the RAM it asked on its side (registered with the read), or the
    // write still in the registered stage that its word hit (a replay: the RAM's word, see above)
    for (genvar p = 0; p < NRP; p++) begin : g_pq
      localparam int SD = sd_of(p);
      logic [LANES-1:0][LSB-1:0] ss;
      logic [LANES-1:0]          bh;
      always_ff @(posedge clk)
        for (int b = 0; b < LANES; b++)
`ifdef OTPU_TMEM_ABL_RPL
          if (1'b0) begin
`else
          if (RPL && rpv[p] && rpe[p][b]) begin
`endif
            ss[b] <= sub_of(rpa[p][b]);
            bh[b] <= 1'b0;
          end else if (pb_en[p][b]) begin
            ss[b] <= sub_of(pb_a[p][b]);
            bh[b] <= pw_v[b][0] && pw_a[b][0] == pb_a[p][b];
          end
      always_comb
`ifdef OTPU_TMEM_ABL_DSEL
        for (int b = 0; b < LANES; b++) pq[p][b] = bh[b] ? pd[b][0] : s_q[b][0][SD];
`else
        for (int b = 0; b < LANES; b++) pq[p][b] = bh[b] ? pd[b][0] : s_q[b][ss[b]][SD];
`endif
`ifndef SYNTHESIS
      initial bh = '0;
`endif
    end
  end else begin : g_nosb
    assign blk_x = '0;
    assign blk_s = '0;
    assign lost = '0;
    assign frz = '0;
    assign rpf = '0;
  end
  for (genvar p = 0; p < NRP; p++) begin : g_port
   if (IMPL != 3 && host_of(p) == p) begin : g_copy
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    if (IMPL == 1 && RQ) begin : g_bq
      // per bank, the asking port on this copy first in priority (ranks are distinct per copy)
      always_comb begin
        b_en = '0; b_a = '0;
        for (int b = 0; b < LANES; b++)
          for (int r = NRP - 1; r >= 0; r--)
            for (int g = 0; g < NRP; g++)
              if (CPY[g] == CPY[p] && int'(PRANK[g]) == r && pb_en[g][b]) begin
                b_en[b] = 1'b1;
                b_a[b] = pb_a[g][b];
              end
      end
    end else if (IMPL == 1) begin : g_bk
      always_comb begin
        b_en = '0; b_a = '0;
        for (int g = 0; g < NRP; g++)
          if (CPY[g] == CPY[p])
            for (int b = 0; b < LANES; b++)
              if (pb_en[g][b]) begin
                b_en[b] = 1'b1;
                b_a[b] = b_a[b] | pb_a[g][b];
              end
      end
    end else if (p == SH_HOST && SH_MASK != '0) begin : g_sh
      always_comb begin
        logic take;
        b_en = pb_en[p]; b_a = pb_a[p];
        take = |r_req[p];
        for (int g = 0; g < NRP; g++)
          if (SH_MASK[g] && !take && |r_req[g]) begin
            b_en = pb_en[g]; b_a = pb_a[g]; take = 1'b1;
          end
      end
    end else begin : g_own
      assign b_en = pb_en[p];
      assign b_a = pb_a[p];
    end
    logic [LANES-1:0][31:0]    q, qb;
    logic [LANES-1:0][WPB-1:0] bh;      // the read hit these writes, still registered
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [31:0] mem [BD];               // IMPL 0's (IMPL 1: unused, the RAM cell holds them)
      if (IMPL == 1) begin : g_ram
        logic [31:0] wd;
`ifdef SYNTHESIS
        assign wd = pw_d[b][0];
`else
        assign wd = pw_d[b][0] ^ fault(p, b, pw_a[b][0]);
`endif
        otpu_ram_1r1w #(.W(32), .N(BD), .STYLE("block")) u_ram (
          .clk, .we(pw_v[b][0]), .wa(pw_a[b][0]), .wd, .re(b_en[b]), .ra(b_a[b]), .rd(q[b]));
      end else begin : g_inf
        always_ff @(posedge clk) begin
          if (b_en[b]) q[b] <= mem[b_a[b]];
          for (int w = 0; w < WPB; w++)
`ifdef SYNTHESIS
            if (pw_v[b][w]) mem[pw_a[b][w]] <= pw_d[b][w];
`else
            if (pw_v[b][w]) mem[pw_a[b][w]] <= pw_d[b][w] ^ fault(p, b, pw_a[b][w]);
`endif
        end
      end
      always_ff @(posedge clk)
        if (b_en[b])
          for (int w = 0; w < WPB; w++) bh[b][w] <= pw_v[b][w] && pw_a[b][w] == b_a[b];
      // the last hit wins, as in the write loop
      always_comb begin
        qb[b] = q[b];
        for (int w = 0; w < WPB; w++) if (bh[b][w]) qb[b] = pd[b][w];
      end
`ifndef SYNTHESIS
      initial for (int i = 0; i < BD; i++) mem[i] = '0;
      initial bh[b] = '0;
      // the dump: DP's words, then the others against them
      if (IMPL == 1) begin : g_dump1
        always @(posedge clk) begin
          if (dump && p == DP)
            for (int r = 0; r < BD; r++) img[r * LANES + b] = g_ram.u_ram.g_bram.mem[r];
          if (dump_q && p != DP)
            for (int r = 0; r < BD; r++)
              if (g_ram.u_ram.g_bram.mem[r] != img[r * LANES + b])
                $fatal(1, "TMEM%0d: copy %0d's word %0d is %h, copy %0d's %h", SID, p,
                       r * LANES + b, g_ram.u_ram.g_bram.mem[r], DP, img[r * LANES + b]);
        end
      end else begin : g_dump0
        always @(posedge clk) begin
          if (dump && p == DP)
            for (int r = 0; r < BD; r++) img[r * LANES + b] = mem[r];
          if (dump_q && p != DP)
            for (int r = 0; r < BD; r++)
              if (mem[r] != img[r * LANES + b])
                $fatal(1, "TMEM%0d: copy %0d's word %0d is %h, copy %0d's %h", SID, p,
                       r * LANES + b, mem[r], DP, img[r * LANES + b]);
        end
      end
`endif
    end
    assign pq[p] = qb;
   end else if (IMPL != 3) begin : g_guest
    assign pq[p] = pq[host_of(p)];
   end
    wire  [BW-1:0]            rot = r_addr[p][0][BW-1:0];
    wire  [LANES-1:0][31:0]   qb = pq[p];
    // lane data: the bank it read (the crossbar: each lane's bank; a rotator: the run's
    // rotation), fresh the cycle after the read, then held. A rotator keeps the rotation once
    // per lane (rot_q[l], all loaded with the run's): lane l takes bank (l + rot_q[l]) mod LANES,
    // so each copy drives one lane's select and hold, not every lane's (133.33 MHz, 110ec6d /
    // 812bb01: g_port[6].rot_q[2] -> 1 LUT -> held, fanout 513, 6.8 ns of route, +0.152 ns)
    logic [LANES-1:0][BW-1:0] sel;
    logic [LANES-1:0][BW-1:0] rq;         // a rotator's rotation, per lane
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held, lq;
    if (GEN_R[p]) begin : g_xb
      assign rq = '0;
    end else begin : g_rot
      (* keep *) logic [LANES-1:0][BW-1:0] rot_q;
      always_ff @(posedge clk)
        for (int l = 0; l < LANES; l++) if (|r_en[p]) rot_q[l] <= rot;
      assign rq = rot_q;
`ifndef SYNTHESIS
      // the lanes' copies are one rotation (the run's), from the first read on
      bit ld;
      initial ld = 1'b0;
      always @(posedge clk) begin
        if (|r_en[p]) ld <= 1'b1;
        for (int l = 1; l < LANES; l++)
          if (ld && rot_q[l] != rot_q[0])
            $fatal(1, "otpu_tmem: port %0d lane %0d's rotation %0d is not lane 0's %0d", p, l,
                   rot_q[l], rot_q[0]);
      end
`endif
    end
    always_ff @(posedge clk)
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= r_en[p][l] | rpf[p][l];   // (rpf: IMPL 3 RPL's replays)
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
  function automatic bit conflict(input logic [LANES-1:0] en, input logic [LANES-1:0][31:0] a,
                                  input bit is_write);
    for (int i = 0; i < LANES; i++)
      for (int j = i + 1; j < LANES; j++)
        if (en[i] && en[j] && (a[i] % LANES) == (a[j] % LANES) && (is_write || a[i] != a[j]))
          return 1;
    return 0;
  endfunction
  function automatic bit contiguous(input logic [LANES-1:0] en, input logic [LANES-1:0][31:0] a);
    for (int l = 0; l < LANES; l++)
      if (en[l] && a[l] != a[0] + 32'(l)) return 0;
    return 1;
  endfunction
  always @(posedge clk) begin
    for (int p = 0; p < NRP; p++)
      if (IMPL == 0 && SH_MASK[p] && |r_en[p]) begin
        if (|r_req[SH_HOST]) $fatal(1, "TMEM%0d: guest port %0d read beside its host at %0t", SID, p, $time);
        for (int g = 0; g < p; g++)
          if (SH_MASK[g] && |r_req[g]) $fatal(1, "TMEM%0d: guest port %0d read beside port %0d at %0t", SID, p, g, $time);
      end
    for (int p = 0; p < NRP; p++)
      if (conflict(r_en[p], r_addr[p], 0)) $fatal(1, "TMEM%0d read port %0d bank conflict at %0t", SID, p, $time);
    for (int p = 0; p < NWP; p++)
      if (conflict(w_en[p], w_addr[p], 1)) $fatal(1, "TMEM%0d write port %0d bank conflict at %0t", SID, p, $time);
    for (int b = 0; b < LANES; b++) begin
      int n;
      n = 0;
      for (int p = 0; p < NWP; p++)
        for (int i = 0; i < LANES; i++)
          if (w_en[p][i] && w_addr[p][i] % LANES == b) n++;
      if (n > WPB) $fatal(1, "TMEM%0d: %0d writes to bank %0d at %0t", SID, n, b, $time);
    end
    for (int p = 0; p < NRP; p++)
      if (!GEN_R[p] && !contiguous(r_req[p], r_addr[p]))
        $fatal(1, "TMEM%0d read port %0d: lanes not contiguous at %0t", SID, p, $time);
    for (int p = 0; p < NWP; p++)
      if (WPB == 1 && !GEN_W[p] && !contiguous(w_req[p], w_addr[p]))
        $fatal(1, "TMEM%0d write port %0d: lanes not contiguous at %0t", SID, p, $time);
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++)
        if (r_en[p][l] && r_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d read beyond %0d words", SID, WORDS);
    for (int p = 0; p < NWP; p++)
      for (int l = 0; l < LANES; l++)
        if (w_en[p][l] && w_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d write beyond %0d words", SID, WORDS);
  end
  string dir;
  initial void'($value$plusargs("dir=%s", dir));
  always @(posedge clk) if (dump_q) begin
    int fd;
    fd = $fopen($sformatf("%s/tmem_%0d.hex", dir, SID), "w");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%08x\n", img[i]);
    $fclose(fd);
  end
`endif
endmodule
