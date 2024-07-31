`timescale 1us/1ns
// use 1us as timescale?

module top_tb;

  // Inputs
  reg sysclk_p;
  reg sysclk_n;
  reg rstn;

  // Outputs
  wire SEGCLK;
  wire SEGCLR;
  wire SEGDT;
  wire SEGEN;

  // Instantiate the design under test
  top dut (
    .sysclk_p(sysclk_p),
    .sysclk_n(sysclk_n),
    .rst(rstn),

    .SEGCLK(SEGCLK),
    .SEGCLR(SEGCLR),
    .SEGDT(SEGDT),
    .SEGEN(SEGEN)
  );

// the 1 means 1 ns (or 1 us)
  initial begin
    sysclk_p    = 0;
    forever #1 sysclk_p = ~sysclk_p;
  end

  initial begin
    sysclk_n     =   1;
    forever #1 sysclk_n = ~sysclk_n;
  end

  initial begin
    rstn  = 1;
    #100;
    rstn = 0;
    #100000000;
    $stop;
  end
endmodule
