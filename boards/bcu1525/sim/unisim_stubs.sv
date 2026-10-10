// Lint-only stand-ins for the unisim buffers otpu_fpga_top_bcu instantiates (no behaviour): `make
// lint` checks the top offline; Vivado builds with the real primitives.
module IBUFDS (input logic I, input logic IB, output logic O);
  assign O = I;
endmodule

module BUFG (input logic I, output logic O);
  assign O = I;
endmodule

module IOBUF (inout wire IO, input logic I, input logic T, output logic O);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
