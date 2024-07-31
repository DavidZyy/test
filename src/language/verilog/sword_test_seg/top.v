module moduleName (
    input   sysclk_p,
    // input   sysclk_n,
    input   rstn,
    input   [15:0]  gpio_l6sw_tri_i,
    output  SEGCLK,
    output  SEGCLR,
    /* DT = data ? */
    output  SEGDT,
    output  SEGEN 
);
    
wire    clk200m;
assign  clk200m =   sysclk_p;

reg [19:0]  clkdiv;
always@(posedge clk200m)
    clkdiv<=clkdiv+1;

// IBUFDS  inst_clk(
// .I(sysclk_p),
// .IB(sysclk_n),
// 
// .O(clk200m)
// );

wire [31:0] num;
assign num[15:0] = gpio_l6sw_tri_i[15:0];
assign num[31:16] = gpio_l6sw_tri_i[15:0];

wire [7:0] dot;
assign  dot=8'b0;
wire [63:0] seg;

hex2seg inst_7seg7(num[31:28],dot[7],seg[63:56]);
hex2seg inst_7seg6(num[27:24],dot[6],seg[55:48]);
hex2seg inst_7seg5(num[23:20],dot[5],seg[47:40]);
hex2seg inst_7seg4(num[19:16],dot[4],seg[39:32]);

hex2seg inst_7seg3(num[15:12],dot[3],seg[31:24]);
hex2seg inst_7seg2(num[11:8],dot[2],seg[23:16]);
hex2seg inst_7seg1(num[7:4],dot[1],seg[15:8]);
hex2seg inst_7seg0(num[3:0],dot[0],seg[7:0]);

SEG7P2S #(
.DATA_BITS(64),//data length
.DATA_COUNT_BITS(6),//data shift bits
.DIR(0)//Shift direction
)
inst_7seg(
.clk(clkdiv[1]),//parallel to serial
.rst(1'b0),
.Start(clkdiv[16]),
.PData(seg),

.s_clk(SEGCLK),
.s_clrn(SEGCLR),
.sout(SEGDT),
.EN(SEGEN)
);

endmodule //moduleName
