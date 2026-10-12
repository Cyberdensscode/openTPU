// TMEM against a reference model under random traffic. Each cycle every port requests a random
// lane set (distinct banks within a port), a random subset of units is granted (writers only
// while their banks stay free, as the slice's arbiter does), and the reads aim mostly at words
// written in the last two cycles, so the registered-write bypass is exercised. A read returns
// the word as of its own cycle (writes of the same cycle are not seen) the cycle after it is
// granted, and the lane holds that value until its next granted read. IMPL 1 (the banked TMEM):
// ports 0 and 1 share copy 0, port 2 has copy 1, and a read port is granted only while its banks
// on its copy are free (as the slice's banked arbiter does). IMPL 3 (one copy, NSB RAMs a bank at
// address bit SBP): a read port is granted only while no port before it in rank asks one of its
// RAMs for another word (blk_x / blk_s, as the slice's grant does). IMPL 3 RPL (replays): the
// grants ignore the RAMs; a read that lost (lost) is read again while its group holds (frz: no
// request from the group's ports), and its lanes show the word of the first read's cycle once a
// replay of it wins (the cycle a read or replay loses, the port's lanes are not checked); SBH
// hashes the sub-bank. TDP: true dual-port RAMs, port 2 on their port A with the writes, ports 0
// and 1 on port B.
module tb_tmem #(parameter int IMPL = 0, parameter int NSB = 4, parameter int SBP = 2,
                 parameter int SBH = 0, parameter bit RPL = 1'b0, parameter bit TDP = 1'b0);
  localparam int WORDS = 256, LANES = 4, NRP = 3, NWP = 2, WPB = 1, CYCLES = 20000;
  localparam int BW = $clog2(LANES);
  localparam int NCP = 2;
  localparam logic [NRP-1:0][2:0] CPY = {3'd1, 3'd0, 3'd0};

  logic clk = 0;
  logic [NRP-1:0][LANES-1:0]       r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       w_en, w_req;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;

  logic [NRP-1:0] blk_x, blk_s, lost, frz;
  localparam logic [NRP-1:0][2:0] GRP = {3'd1, 3'd1, 3'd0};
  otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB), .IMPL(IMPL),
              .NCP(NCP), .CPY(CPY), .NSB(NSB), .SBP(SBP), .PRANK({3'd2, 3'd1, 3'd0}),
              .GRP(GRP), .SBH(SBH), .RPL(RPL), .TOPP(0), .TDP(TDP), .SIDE(3'b100)) dut (
    .clk, .r_en, .r_req, .r_addr, .r_data, .w_en, .w_req, .w_gnt, .w_addr, .w_data,
    .blk_x, .blk_s, .lost, .frz, .dump(1'b0));

  logic [31:0] ref_mem [WORDS];
  logic [31:0] expect_q [NRP][LANES];
  bit          valid_q [NRP][LANES];
  int unsigned recent [8];              // addresses written in the last cycles
  int unsigned last_wrote [$];          // the previous cycle's writes (still registered in TMEM)
  int          hits = 0, checks = 0, blocked = 0, replays = 0;
  logic [NRP-1:0] lostc;
  logic [NRP-1:0][LANES-1:0]       pend_v = '0;
  logic [NRP-1:0][LANES-1:0][31:0] pend_a;

  function automatic int unsigned pick_addr(int bank);
    int unsigned a;
    if ($urandom_range(3) != 0) begin
      a = recent[$urandom_range(7)];
      a = (a & ~(LANES - 1)) | bank;    // same row neighbourhood, forced to the lane's bank
    end else
      a = ($urandom_range(WORDS / LANES - 1) * LANES) | bank;
    return a % WORDS;
  endfunction

  initial begin
    for (int i = 0; i < WORDS; i++) ref_mem[i] = '0;
    for (int i = 0; i < 8; i++) recent[i] = 0;
    for (int p = 0; p < NRP; p++) for (int l = 0; l < LANES; l++) valid_q[p][l] = 0;
    r_en = '0; r_req = '0; r_addr = '0; w_en = '0; w_req = '0; w_gnt = '0; w_addr = '0; w_data = '0;

    for (int cyc = 0; cyc < CYCLES; cyc++) begin
      logic [LANES-1:0] busy;
      logic [NCP-1:0][LANES-1:0] rbusy;
      int unsigned      wrote [$];
      // ---- reads
      rbusy = '0;
      for (int p = 0; p < NRP; p++) begin
        int perm [LANES];
        bit g;
        logic [LANES-1:0] banks;
        for (int l = 0; l < LANES; l++) perm[l] = l;
        perm.shuffle();
        banks = '0;
        for (int l = 0; l < LANES; l++) begin
          r_req[p][l] = $urandom_range(2) != 0;
          r_addr[p][l] = pick_addr(perm[l]);
          if (r_req[p][l]) banks[perm[l]] = 1'b1;
        end
        g = $urandom_range(3) != 0 && (IMPL != 1 || (banks & rbusy[CPY[p]]) == '0);
        if (g && IMPL == 1) rbusy[CPY[p]] |= banks;
        if (RPL)                       // a group replaying a lost read holds: no request
          for (int q = 0; q < NRP; q++)
            if (frz[q] && GRP[q] == GRP[p]) begin g = 0; r_req[p] = '0; end
        for (int l = 0; l < LANES; l++) r_en[p][l] = g && r_req[p][l];
      end
      // ---- writes: each port's lanes take a random permutation of banks
      busy = '0;
      for (int p = 0; p < NWP; p++) begin
        int perm [LANES];
        logic [LANES-1:0] banks;
        for (int l = 0; l < LANES; l++) perm[l] = l;
        perm.shuffle();
        banks = '0;
        for (int l = 0; l < LANES; l++) begin
          w_req[p][l] = $urandom_range(2) != 0;
          w_addr[p][l] = pick_addr(perm[l]);
          // RPL: no write to a word read this cycle or awaiting a replay (the slice's
          // scoreboard keeps a word a unit reads from other units' writes until it is done)
          if (RPL)
            for (int q = 0; q < NRP; q++)
              for (int k = 0; k < LANES; k++)
                if ((r_req[q][k] && r_addr[q][k] == w_addr[p][l]) ||
                    (pend_v[q][k] && pend_a[q][k] == w_addr[p][l])) w_req[p][l] = 1'b0;
          w_data[p][l] = $urandom;
          if (w_req[p][l]) banks[perm[l]] = 1'b1;
        end
        w_gnt[p] = $urandom_range(3) != 0 && (banks & busy) == '0;
        if (w_gnt[p]) busy |= banks;
        w_en[p] = w_gnt[p] ? w_req[p] : '0;
      end
      #1;
      if (IMPL == 3 && !RPL) begin     // the requests settled: no read of a RAM another port took
        for (int p = 0; p < NRP; p++)
          if (blk_x[p] || blk_s[p]) begin r_en[p] = '0; blocked++; end
        #1;
      end
      lostc = (IMPL == 3 && RPL) ? lost : '0;
      for (int p = 0; p < NRP; p++) begin          // the words awaiting a replay
        if (frz[p] && !lostc[p]) pend_v[p] = '0;
        if (lostc[p] && !frz[p])
          for (int l = 0; l < LANES; l++) begin
            pend_v[p][l] = r_en[p][l];
            pend_a[p][l] = r_addr[p][l];
          end
      end
      for (int p = 0; p < NRP; p++) if (lostc[p]) blocked++;
      for (int p = 0; p < NRP; p++) if (frz[p]) replays++;
      // ---- the reads see the memory before this cycle's writes
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++)
          if (r_en[p][l]) begin
            foreach (last_wrote[i])
              if (last_wrote[i] == r_addr[p][l]) begin hits++; break; end
            expect_q[p][l] = ref_mem[r_addr[p][l]];
            valid_q[p][l] = 1;
          end
      for (int p = 0; p < NWP; p++)
        for (int l = 0; l < LANES; l++)
          if (w_en[p][l]) begin
            ref_mem[w_addr[p][l]] = w_data[p][l];
            wrote.push_back(w_addr[p][l]);
          end
      for (int i = 0; i < wrote.size() && i < 8; i++) recent[(cyc * 3 + i) % 8] = wrote[i];
      last_wrote = wrote;
      clk = 1; #1; clk = 0; #1;
      // ---- every lane shows its last granted read
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++)
          if (valid_q[p][l] && !lostc[p]) begin
            checks++;
            if (r_data[p][l] !== expect_q[p][l]) begin
              $display("FAIL cycle %0d port %0d lane %0d: got %08x want %08x", cyc, p, l,
                       r_data[p][l], expect_q[p][l]);
              $fatal(1);
            end
          end
    end
    if (hits < 1000) $fatal(1, "only %0d reads of the previous cycle's writes", hits);
    if (IMPL == 3 && blocked < 1000) $fatal(1, "only %0d blocked reads", blocked);
    if (RPL && replays < 1000) $fatal(1, "only %0d replays", replays);
    $display("PASS checks=%0d bypass_reads=%0d blocked=%0d replays=%0d", checks, hits, blocked,
             replays);
    $finish;
  end
endmodule
