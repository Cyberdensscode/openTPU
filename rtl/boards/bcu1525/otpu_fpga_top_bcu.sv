// FPGA top for the BCU1525 (VCU1525 build: xcvu9p-fsgd2104-2L-e; docs/bcu1525.md): the block
// design otpu_bd (XDMA Gen3, the core clock, the SYSMON, the control interconnect and XDMA's DMA
// split onto the channels: boards/bcu1525/vivado/bd.tcl), one DDR4 LiteDRAM core per DIMM
// (otpu_ld4, boards/bcu1525/litedram/: USPDDRPHY RDIMM x4 at DDR4-1600, ECC, its own VexRiscv
// calibrating it at reset) and otpu_bcu_sys: the accelerator (otpu_board, S = 2) and XDMA's DMA,
// meeting in one otpu_mem_ch per channel.
//
// Channels: board channel j = 2s + c (slice s's channel c; the host's channel_map) is DIMM
// dimm(j) = C0, C1, C3, C2 for j = 0..3: slice 0 on C0 (SLR0) and C1 (SLR1), slice 1 on C3 (SLR2)
// and C2 (SLR1), so each slice has one channel in its own SLR and one a crossing away
// (constraints/bcu1525_floorplan.xdc). The DIMMs' CSRs keep their DIMM's number (BAR0 0x40000 +
// d * 0x10000), so a channel's controller is found by its pins.
//
// Clocks: each DIMM's 300 MHz clock (clk300_<d>, the bank's LVDS input) on one global buffer feeds
// that DIMM's core (its MMCM makes sys_clk, 200 MHz, and the PHY's 800 MHz and IDELAYCTRL's 400
// MHz); clk300_1 (SLR1, with the PCIe block) also feeds the block design's MMCM (core_clk).
// xdma_aclk is XDMA's (250 MHz). otpu_mem_ch crosses between core_clk, xdma_aclk and its DIMM's
// sys_clk.
// Resets: core_rst from the block design (MMCM lock, PCIe PERST#), xrst from XDMA's axi_aresetn;
// the DDR4 cores have none (their MMCMs' lock resets them), so they calibrate once per
// configuration, by themselves (HOSTCAL 0). Each core's CSR port runs on core_clk, reset with its
// sys reset brought into core_clk (as the YPCB top: the two sides of its crossing reset
// together), behind otpu_axil_iso against the SmartConnect's reset.
// The RDIMM pins the single-rank core does not drive: par 0 (the RCD's parity check is off after
// reset), the other ranks' chip selects high, cke1 / odt1 low. The UART is DIMM C1's core's (its
// BIOS log: the calibration).
// The parameters are the target's (targets/bcu1525.toml: opentpu.target rtl_params; run_vivado.sh
// passes them, the defaults are its values: tests/test_target.py).
module otpu_fpga_top_bcu #(
  parameter int S          = 2,
  parameter int D          = 128,
  parameter int MCOLS      = 4,
  parameter int ACT_ROWS   = MCOLS,
  parameter int ACT_BLOCKS = 128,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int RPB        = 8 * LANES,
  parameter int WPB        = 1,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = 2,
  parameter int ULANES     = 8,
  parameter int TRACE_DEPTH = 16384,
  parameter int PQ_WIN     = 1024,
  parameter bit CHASH      = 1'b1,
  parameter bit DSTEP      = 1'b1,
  parameter int CORE_KHZ   = 100000,        // core_clk as the block design makes it
  parameter logic [31:0] BUILD_ID = 32'h0,  // the git commit (BUILD_ID register)
  parameter int DDR_MTS    = 1600,          // the DDR4 data rate the cores run (DDR_MTS register)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int PLATFORM   = 2,             // targets/bcu1525.toml (CFG_PLATFORM)
  parameter logic [31:0] ARCH_HASH = 32'h0, // its arch_hash (CFG_HASH; run_vivado.sh)
  parameter int PCIE_LANES = 16             // Gen3 x16 (or x8: bd.tcl)
) (
  // ---- board
  input  logic        clk300_0_p,
  input  logic        clk300_0_n,
  input  logic        clk300_1_p,
  input  logic        clk300_1_n,
  input  logic        clk300_2_p,
  input  logic        clk300_2_n,
  input  logic        clk300_3_p,
  input  logic        clk300_3_n,
  output logic [2:0]  led,
  input  logic        uart_rx,
  output logic        uart_tx,
  inout  wire         i2c_scl,
  inout  wire         i2c_sda,
  output logic        i2c_mux_rst_n,
  // ---- PCIe Gen3 (the GT pins are placed by XDMA's select_quad: GTY quads 224-227)
  input  logic        pcie_refclk_p,
  input  logic        pcie_refclk_n,
  input  logic        pcie_perst_n,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp,
  output logic [PCIE_LANES-1:0] pcie_mgt_txn,
  // ---- the four DDR4 RDIMMs (C0 .. C3; constraints/ddr4_c<d>.pins.xdc; par, cke1, odt1 and
  // cs_hi: the RDIMM pins the core does not drive)
  output logic [13:0] ddr4_c0_a,
  output logic        ddr4_c0_we_n,
  output logic        ddr4_c0_cas_n,
  output logic        ddr4_c0_ras_n,
  output logic        ddr4_c0_act_n,
  output logic [1:0]  ddr4_c0_ba,
  output logic [1:0]  ddr4_c0_bg,
  output logic        ddr4_c0_cke,
  output logic        ddr4_c0_odt,
  output logic        ddr4_c0_cs_n,
  output logic        ddr4_c0_reset_n,
  output logic        ddr4_c0_clk_p,
  output logic        ddr4_c0_clk_n,
  output logic        ddr4_c0_par,
  output logic        ddr4_c0_cke1,
  output logic        ddr4_c0_odt1,
  output logic [2:0]  ddr4_c0_cs_hi,
  inout  wire  [71:0] ddr4_c0_dq,
  inout  wire  [17:0] ddr4_c0_dqs_p,
  inout  wire  [17:0] ddr4_c0_dqs_n,
  output logic [13:0] ddr4_c1_a,
  output logic        ddr4_c1_we_n,
  output logic        ddr4_c1_cas_n,
  output logic        ddr4_c1_ras_n,
  output logic        ddr4_c1_act_n,
  output logic [1:0]  ddr4_c1_ba,
  output logic [1:0]  ddr4_c1_bg,
  output logic        ddr4_c1_cke,
  output logic        ddr4_c1_odt,
  output logic        ddr4_c1_cs_n,
  output logic        ddr4_c1_reset_n,
  output logic        ddr4_c1_clk_p,
  output logic        ddr4_c1_clk_n,
  output logic        ddr4_c1_par,
  output logic        ddr4_c1_cke1,
  output logic        ddr4_c1_odt1,
  output logic [2:0]  ddr4_c1_cs_hi,
  inout  wire  [71:0] ddr4_c1_dq,
  inout  wire  [17:0] ddr4_c1_dqs_p,
  inout  wire  [17:0] ddr4_c1_dqs_n,
  output logic [13:0] ddr4_c2_a,
  output logic        ddr4_c2_we_n,
  output logic        ddr4_c2_cas_n,
  output logic        ddr4_c2_ras_n,
  output logic        ddr4_c2_act_n,
  output logic [1:0]  ddr4_c2_ba,
  output logic [1:0]  ddr4_c2_bg,
  output logic        ddr4_c2_cke,
  output logic        ddr4_c2_odt,
  output logic        ddr4_c2_cs_n,
  output logic        ddr4_c2_reset_n,
  output logic        ddr4_c2_clk_p,
  output logic        ddr4_c2_clk_n,
  output logic        ddr4_c2_par,
  output logic        ddr4_c2_cke1,
  output logic        ddr4_c2_odt1,
  output logic [2:0]  ddr4_c2_cs_hi,
  inout  wire  [71:0] ddr4_c2_dq,
  inout  wire  [17:0] ddr4_c2_dqs_p,
  inout  wire  [17:0] ddr4_c2_dqs_n,
  output logic [13:0] ddr4_c3_a,
  output logic        ddr4_c3_we_n,
  output logic        ddr4_c3_cas_n,
  output logic        ddr4_c3_ras_n,
  output logic        ddr4_c3_act_n,
  output logic [1:0]  ddr4_c3_ba,
  output logic [1:0]  ddr4_c3_bg,
  output logic        ddr4_c3_cke,
  output logic        ddr4_c3_odt,
  output logic        ddr4_c3_cs_n,
  output logic        ddr4_c3_reset_n,
  output logic        ddr4_c3_clk_p,
  output logic        ddr4_c3_clk_n,
  output logic        ddr4_c3_par,
  output logic        ddr4_c3_cke1,
  output logic        ddr4_c3_odt1,
  output logic [2:0]  ddr4_c3_cs_hi,
  inout  wire  [71:0] ddr4_c3_dq,
  inout  wire  [17:0] ddr4_c3_dqs_p,
  inout  wire  [17:0] ddr4_c3_dqs_n
);
  localparam int NC = 2 * S;
  // board channel j's DIMM (see the header)
  function automatic int dimm(int j);
    return j == 2 ? 3 : j == 3 ? 2 : j;
  endfunction
`ifndef SYNTHESIS
  initial if (S != 2) $fatal(1, "otpu_fpga_top_bcu: S = 2 (four DDR4 channels, two per slice)");
`endif

  // ---- clocks: each DIMM's 300 MHz on one global buffer
  logic [3:0] c300_ib, c300;
  IBUFDS u_ib300_0 (.I(clk300_0_p), .IB(clk300_0_n), .O(c300_ib[0]));
  IBUFDS u_ib300_1 (.I(clk300_1_p), .IB(clk300_1_n), .O(c300_ib[1]));
  IBUFDS u_ib300_2 (.I(clk300_2_p), .IB(clk300_2_n), .O(c300_ib[2]));
  IBUFDS u_ib300_3 (.I(clk300_3_p), .IB(clk300_3_n), .O(c300_ib[3]));
  for (genvar d = 0; d < 4; d++) begin : g_bufg
    BUFG u_bufg (.I(c300_ib[d]), .O(c300[d]));
  end

  // ---- the block design
  logic        core_clk, core_rstn, xdma_aclk, xdma_aresetn, pcie_link_up;
  logic [9:0]  device_temp;                 // SYSMONE4 die-temperature code (core_clk)
  // control (AXI4-Lite, core_clk)
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;
  // the DIMMs' CSRs (AXI4-Lite, core_clk), [d]
  logic [3:0][31:0] mc_awaddr, mc_araddr, mc_wdata, mc_rdata;
  logic [3:0][3:0]  mc_wstrb;
  logic [3:0][1:0]  mc_bresp, mc_rresp;
  logic [3:0] mc_awvalid, mc_awready, mc_wvalid, mc_wready, mc_bvalid, mc_bready;
  logic [3:0] mc_arvalid, mc_arready, mc_rvalid, mc_rready;
  // XDMA's DMA per board channel (AXI4, 128 bits, xdma_aclk), [j]
  logic [NC-1:0][32:0]  dm_awaddr, dm_araddr;
  logic [NC-1:0][7:0]   dm_awlen, dm_arlen;
  logic [NC-1:0][127:0] dm_wdata, dm_rdata;
  logic [NC-1:0][15:0]  dm_wstrb;
  logic [NC-1:0][1:0]   dm_bresp, dm_rresp;
  logic [NC-1:0] dm_awvalid, dm_awready, dm_wvalid, dm_wready, dm_wlast, dm_bvalid, dm_bready;
  logic [NC-1:0] dm_arvalid, dm_arready, dm_rvalid, dm_rready, dm_rlast;

`define BD_LITE(P, S) \
    .P``_awaddr(S``_awaddr), .P``_awvalid(S``_awvalid), .P``_awready(S``_awready), \
    .P``_wdata(S``_wdata), .P``_wstrb(S``_wstrb), .P``_wvalid(S``_wvalid), \
    .P``_wready(S``_wready), .P``_bresp(S``_bresp), .P``_bvalid(S``_bvalid), \
    .P``_bready(S``_bready), .P``_araddr(S``_araddr), .P``_arvalid(S``_arvalid), \
    .P``_arready(S``_arready), .P``_rdata(S``_rdata), .P``_rresp(S``_rresp), \
    .P``_rvalid(S``_rvalid), .P``_rready(S``_rready)
`define BD_MC(D) \
    .M_AXI_MEMCAL``D``_awaddr(mc_awaddr[D]), .M_AXI_MEMCAL``D``_awvalid(mc_awvalid[D]), \
    .M_AXI_MEMCAL``D``_awready(mc_awready[D]), .M_AXI_MEMCAL``D``_wdata(mc_wdata[D]), \
    .M_AXI_MEMCAL``D``_wstrb(mc_wstrb[D]), .M_AXI_MEMCAL``D``_wvalid(mc_wvalid[D]), \
    .M_AXI_MEMCAL``D``_wready(mc_wready[D]), .M_AXI_MEMCAL``D``_bresp(mc_bresp[D]), \
    .M_AXI_MEMCAL``D``_bvalid(mc_bvalid[D]), .M_AXI_MEMCAL``D``_bready(mc_bready[D]), \
    .M_AXI_MEMCAL``D``_araddr(mc_araddr[D]), .M_AXI_MEMCAL``D``_arvalid(mc_arvalid[D]), \
    .M_AXI_MEMCAL``D``_arready(mc_arready[D]), .M_AXI_MEMCAL``D``_rdata(mc_rdata[D]), \
    .M_AXI_MEMCAL``D``_rresp(mc_rresp[D]), .M_AXI_MEMCAL``D``_rvalid(mc_rvalid[D]), \
    .M_AXI_MEMCAL``D``_rready(mc_rready[D])
`define BD_DMA(J) \
    .M_AXI_DMA``J``_awaddr(dm_awaddr[J]), .M_AXI_DMA``J``_awlen(dm_awlen[J]), \
    .M_AXI_DMA``J``_awvalid(dm_awvalid[J]), .M_AXI_DMA``J``_awready(dm_awready[J]), \
    .M_AXI_DMA``J``_wdata(dm_wdata[J]), .M_AXI_DMA``J``_wstrb(dm_wstrb[J]), \
    .M_AXI_DMA``J``_wlast(dm_wlast[J]), .M_AXI_DMA``J``_wvalid(dm_wvalid[J]), \
    .M_AXI_DMA``J``_wready(dm_wready[J]), .M_AXI_DMA``J``_bresp(dm_bresp[J]), \
    .M_AXI_DMA``J``_bvalid(dm_bvalid[J]), .M_AXI_DMA``J``_bready(dm_bready[J]), \
    .M_AXI_DMA``J``_araddr(dm_araddr[J]), .M_AXI_DMA``J``_arlen(dm_arlen[J]), \
    .M_AXI_DMA``J``_arvalid(dm_arvalid[J]), .M_AXI_DMA``J``_arready(dm_arready[J]), \
    .M_AXI_DMA``J``_rdata(dm_rdata[J]), .M_AXI_DMA``J``_rresp(dm_rresp[J]), \
    .M_AXI_DMA``J``_rlast(dm_rlast[J]), .M_AXI_DMA``J``_rvalid(dm_rvalid[J]), \
    .M_AXI_DMA``J``_rready(dm_rready[J])
  otpu_bd_wrapper u_bd (
    .clk300_in(c300[1]),
    .pcie_refclk_clk_p(pcie_refclk_p), .pcie_refclk_clk_n(pcie_refclk_n),
    .pcie_perstn(pcie_perst_n),
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .xdma_aclk, .xdma_aresetn, .device_temp,
    `BD_LITE(M_AXI_CTL, ctl),
    `BD_MC(0), `BD_MC(1), `BD_MC(2), `BD_MC(3),
    `BD_DMA(0), `BD_DMA(1), `BD_DMA(2), `BD_DMA(3));
`undef BD_LITE
`undef BD_MC
`undef BD_DMA

  // resets: synchronous, active high, one register each (otpu_mem_ch crosses them registered)
  logic core_rst, xrst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;
  always_ff @(posedge xdma_aclk) xrst <= !xdma_aresetn;
  // the SmartConnect's reset (axi_aresetn) in core_clk, where its control masters are: the masters'
  // side of otpu_axil_iso (control registers and CSRs). Two flip-flops; the reset lasts many cycles
  (* ASYNC_REG = "TRUE" *) logic lrs1 = 1'b1, lrs2 = 1'b1;
  always_ff @(posedge core_clk) begin lrs1 <= xrst; lrs2 <= lrs1; end

  // ---- the DDR4 cores, [d] = DIMM C<d>
  logic [3:0]             d_sys_clk, d_sys_rst, d_ready, d_ded, d_tx;
  logic [3:0][1:0]        d_cmd_valid, d_cmd_ready, d_cmd_we, d_wdata_valid, d_wdata_ready;
  logic [3:0][1:0]        d_rdata_valid;
  logic [3:0][1:0][27:0]  d_cmd_addr;
  logic [3:0][1:0][511:0] d_wdata_data, d_rdata_data;
  logic [3:0][1:0][63:0]  d_wdata_we;
  // each core's CSR port: its sys reset in core_clk (two flip-flops), behind otpu_axil_iso
  (* ASYNC_REG = "TRUE" *) logic [3:0] ld_crs1 = '1, ld_crs2 = '1;
  always_ff @(posedge core_clk) begin ld_crs1 <= d_sys_rst; ld_crs2 <= ld_crs1; end
  logic [3:0][15:0] lc_awaddr, lc_araddr;
  logic [3:0][31:0] lc_wdata, lc_rdata;
  logic [3:0][3:0]  lc_wstrb;
  logic [3:0][1:0]  lc_bresp, lc_rresp;
  logic [3:0] lc_awvalid, lc_awready, lc_wvalid, lc_wready, lc_bvalid, lc_bready;
  logic [3:0] lc_arvalid, lc_arready, lc_rvalid, lc_rready;
  for (genvar d = 0; d < 4; d++) begin : g_csr
    otpu_axil_iso #(.AW(16)) u_iso (
      .clk(core_clk), .m_rst(lrs2), .s_rst(ld_crs2[d]),
      .m_awaddr(mc_awaddr[d][15:0]), .m_awvalid(mc_awvalid[d]), .m_awready(mc_awready[d]),
      .m_wdata(mc_wdata[d]), .m_wstrb(mc_wstrb[d]), .m_wvalid(mc_wvalid[d]),
      .m_wready(mc_wready[d]), .m_bresp(mc_bresp[d]), .m_bvalid(mc_bvalid[d]),
      .m_bready(mc_bready[d]), .m_araddr(mc_araddr[d][15:0]), .m_arvalid(mc_arvalid[d]),
      .m_arready(mc_arready[d]), .m_rdata(mc_rdata[d]), .m_rresp(mc_rresp[d]),
      .m_rvalid(mc_rvalid[d]), .m_rready(mc_rready[d]),
      .s_awaddr(lc_awaddr[d]), .s_awvalid(lc_awvalid[d]), .s_awready(lc_awready[d]),
      .s_wdata(lc_wdata[d]), .s_wstrb(lc_wstrb[d]), .s_wvalid(lc_wvalid[d]),
      .s_wready(lc_wready[d]), .s_bresp(lc_bresp[d]), .s_bvalid(lc_bvalid[d]),
      .s_bready(lc_bready[d]), .s_araddr(lc_araddr[d]), .s_arvalid(lc_arvalid[d]),
      .s_arready(lc_arready[d]), .s_rdata(lc_rdata[d]), .s_rresp(lc_rresp[d]),
      .s_rvalid(lc_rvalid[d]), .s_rready(lc_rready[d]));
  end

`define LD4(N) \
  otpu_ld4 u_ld``N ( \
    .clk300(c300[N]), .rst(1'b0), .sys_clk(d_sys_clk[N]), .sys_rst(d_sys_rst[N]), \
    .ctl_clk(core_clk), .ctl_rst(ld_crs2[N]), \
    .ctl_awvalid(lc_awvalid[N]), .ctl_awready(lc_awready[N]), .ctl_awaddr(lc_awaddr[N]), \
    .ctl_wvalid(lc_wvalid[N]), .ctl_wready(lc_wready[N]), .ctl_wdata(lc_wdata[N]), \
    .ctl_wstrb(lc_wstrb[N]), .ctl_bvalid(lc_bvalid[N]), .ctl_bready(lc_bready[N]), \
    .ctl_bresp(lc_bresp[N]), .ctl_arvalid(lc_arvalid[N]), .ctl_arready(lc_arready[N]), \
    .ctl_araddr(lc_araddr[N]), .ctl_rvalid(lc_rvalid[N]), .ctl_rready(lc_rready[N]), \
    .ctl_rdata(lc_rdata[N]), .ctl_rresp(lc_rresp[N]), \
    .c0_cmd_valid(d_cmd_valid[N][0]), .c0_cmd_ready(d_cmd_ready[N][0]), \
    .c0_cmd_we(d_cmd_we[N][0]), .c0_cmd_addr(d_cmd_addr[N][0]), \
    .c0_wdata_valid(d_wdata_valid[N][0]), .c0_wdata_ready(d_wdata_ready[N][0]), \
    .c0_wdata_data(d_wdata_data[N][0]), .c0_wdata_we(d_wdata_we[N][0]), \
    .c0_rdata_valid(d_rdata_valid[N][0]), .c0_rdata_ready(1'b1), \
    .c0_rdata_data(d_rdata_data[N][0]), .c0_ready(d_ready[N]), .c0_ded(d_ded[N]), \
    .c0b_cmd_valid(d_cmd_valid[N][1]), .c0b_cmd_ready(d_cmd_ready[N][1]), \
    .c0b_cmd_we(d_cmd_we[N][1]), .c0b_cmd_addr(d_cmd_addr[N][1]), \
    .c0b_wdata_valid(d_wdata_valid[N][1]), .c0b_wdata_ready(d_wdata_ready[N][1]), \
    .c0b_wdata_data(d_wdata_data[N][1]), .c0b_wdata_we(d_wdata_we[N][1]), \
    .c0b_rdata_valid(d_rdata_valid[N][1]), .c0b_rdata_ready(1'b1), \
    .c0b_rdata_data(d_rdata_data[N][1]), \
    .serial_rx(N == 1 ? uart_rx : 1'b1), .serial_tx(d_tx[N]), \
    .ddram_a(ddr4_c``N``_a), .ddram_we_n(ddr4_c``N``_we_n), .ddram_cas_n(ddr4_c``N``_cas_n), \
    .ddram_ras_n(ddr4_c``N``_ras_n), .ddram_act_n(ddr4_c``N``_act_n), \
    .ddram_ba(ddr4_c``N``_ba), .ddram_bg(ddr4_c``N``_bg), .ddram_cke(ddr4_c``N``_cke), \
    .ddram_odt(ddr4_c``N``_odt), .ddram_cs_n(ddr4_c``N``_cs_n), \
    .ddram_reset_n(ddr4_c``N``_reset_n), .ddram_clk_p(ddr4_c``N``_clk_p), \
    .ddram_clk_n(ddr4_c``N``_clk_n), .ddram_dq(ddr4_c``N``_dq), \
    .ddram_dqs_p(ddr4_c``N``_dqs_p), .ddram_dqs_n(ddr4_c``N``_dqs_n)); \
  assign ddr4_c``N``_par = 1'b0; \
  assign ddr4_c``N``_cs_hi = 3'b111; \
  assign ddr4_c``N``_cke1 = 1'b0; \
  assign ddr4_c``N``_odt1 = 1'b0;
  `LD4(0)
  `LD4(1)
  `LD4(2)
  `LD4(3)
`undef LD4
  assign uart_tx = d_tx[1];

  // ---- board channel j <-> DIMM dimm(j)
  logic [NC-1:0]             calib, ded, uclk, urst;
  logic [NC-1:0][1:0]        c_cmd_valid, c_cmd_ready, c_cmd_we, c_wdata_valid, c_wdata_ready;
  logic [NC-1:0][1:0]        c_rdata_valid;
  logic [NC-1:0][1:0][24:0]  c_cmd_addr;
  logic [NC-1:0][1:0][511:0] c_wdata_data, c_rdata_data;
  logic [NC-1:0][1:0][63:0]  c_wdata_we;
  for (genvar j = 0; j < NC; j++) begin : g_map
    localparam int M = dimm(j);
    assign calib[j] = d_ready[M];             // STATUS CALIB0/1 (sys_clk; otpu_board syncs them)
    assign ded[j]   = d_ded[M];
    assign uclk[j]  = d_sys_clk[M];
    assign urst[j]  = d_sys_rst[M];
    assign d_cmd_valid[M]   = c_cmd_valid[j];
    assign c_cmd_ready[j]   = d_cmd_ready[M];
    assign d_cmd_we[M]      = c_cmd_we[j];
    // the beat in the DIMM's 16 GiB: the 2 GiB a slice addresses are its first
    for (genvar p = 0; p < 2; p++) begin : g_port
      assign d_cmd_addr[M][p] = {3'b000, c_cmd_addr[j][p]};
    end
    assign d_wdata_valid[M] = c_wdata_valid[j];
    assign c_wdata_ready[j] = d_wdata_ready[M];
    assign d_wdata_data[M]  = c_wdata_data[j];
    assign d_wdata_we[M]    = c_wdata_we[j];
    assign c_rdata_valid[j] = d_rdata_valid[M];
    assign c_rdata_data[j]  = d_rdata_data[M];
  end

  // ---- I2C: the board's bus (behind its mux, held out of reset) on bus 0's pins, released unless
  // I2C_CTRL drives a line low; bus 1 and ALERT0 read high (otpu_bcu_sys: CAPS bit2 clear)
  logic [3:0] i2c_lo;
  logic [1:0] i2c_lvl;
  IOBUF u_iob_scl (.IO(i2c_scl), .I(1'b0), .T(!i2c_lo[0]), .O(i2c_lvl[0]));
  IOBUF u_iob_sda (.IO(i2c_sda), .I(1'b0), .T(!i2c_lo[1]), .O(i2c_lvl[1]));
  assign i2c_mux_rst_n = 1'b1;

  // ---- the accelerator and XDMA on the channels
  // the channel's 2 GiB (sc_dma decoded bits 32:31)
  logic [NC-1:0][30:0] dm_awaddr_c, dm_araddr_c;
  for (genvar j = 0; j < NC; j++) begin : g_dma
    assign dm_awaddr_c[j] = dm_awaddr[j][30:0];
    assign dm_araddr_c[j] = dm_araddr[j][30:0];
  end
  logic [2:0] board_led;
  otpu_bcu_sys #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
                 .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
                 .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .TRACE_DEPTH(TRACE_DEPTH),
                 .PQ_WIN(PQ_WIN), .CHASH(CHASH), .DSTEP(DSTEP), .CORE_KHZ(CORE_KHZ),
                 .BUILD_ID(BUILD_ID), .SEQ_WC(SEQ_WC), .DDR_MTS(DDR_MTS), .PLATFORM(PLATFORM),
                 .ARCH_HASH(ARCH_HASH)) u_sys (
    .clk(core_clk), .rst(core_rst), .ctl_mrst(lrs2), .xclk(xdma_aclk), .xrst,
    .calib, .ded, .temp({device_temp, 2'b00}), .led(board_led), .i2c_lo,
    .i2c_pin({3'b111, i2c_lvl}),
    .s_ctl_awaddr(ctl_awaddr[11:0]), .s_ctl_awvalid(ctl_awvalid), .s_ctl_awready(ctl_awready),
    .s_ctl_wdata(ctl_wdata), .s_ctl_wstrb(ctl_wstrb), .s_ctl_wvalid(ctl_wvalid),
    .s_ctl_wready(ctl_wready), .s_ctl_bresp(ctl_bresp), .s_ctl_bvalid(ctl_bvalid),
    .s_ctl_bready(ctl_bready), .s_ctl_araddr(ctl_araddr[11:0]), .s_ctl_arvalid(ctl_arvalid),
    .s_ctl_arready(ctl_arready), .s_ctl_rdata(ctl_rdata), .s_ctl_rresp(ctl_rresp),
    .s_ctl_rvalid(ctl_rvalid), .s_ctl_rready(ctl_rready),
    .x_awvalid(dm_awvalid), .x_awready(dm_awready), .x_awaddr(dm_awaddr_c), .x_awlen(dm_awlen),
    .x_wvalid(dm_wvalid), .x_wready(dm_wready), .x_wdata(dm_wdata), .x_wstrb(dm_wstrb),
    .x_wlast(dm_wlast), .x_bvalid(dm_bvalid), .x_bready(dm_bready), .x_bresp(dm_bresp),
    .x_arvalid(dm_arvalid), .x_arready(dm_arready), .x_araddr(dm_araddr_c), .x_arlen(dm_arlen),
    .x_rvalid(dm_rvalid), .x_rready(dm_rready), .x_rdata(dm_rdata), .x_rresp(dm_rresp),
    .x_rlast(dm_rlast),
    .uclk, .urst,
    .c_cmd_valid, .c_cmd_ready, .c_cmd_we, .c_cmd_addr, .c_wdata_valid, .c_wdata_ready,
    .c_wdata_data, .c_wdata_we, .c_rdata_valid, .c_rdata_data);

  // LEDs: led[0] heartbeat, led[1] PCIe link up and every channel calibrated (by its core), led[2]
  // the accelerator runs or halted cleanly
  assign led = {board_led[2] | board_led[1], pcie_link_up & (&d_ready), board_led[0]};
endmodule
