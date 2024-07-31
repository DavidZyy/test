
module top (
    input   sysclk_p,
    input   sysclk_n,
    input   rst,
    input   btn_clk,

    output  [7:0] leds,
    output  SEGCLK,
    output  SEGCLR,
    /* DT = data ? */
    output  SEGDT,
    output  SEGEN
    // output  led
);

    wire    clk200m;
    reg [31:0]  clkdiv;

    IBUFDS  inst_clk(
        .I(sysclk_p),
        .IB(sysclk_n),

        .O(clk200m)
    );

    always@(posedge clk200m)
        clkdiv <= clkdiv+1;

// always@(posedge clk200m or posedge rst)
// 	if(rst) begin
// 		clkdiv <= 0;
// 	end
// 	else begin
//     	clkdiv <= clkdiv+1;
// 	end


    reg [31:0] num;

    /* use button as clk */
    // always @(posedge clkdiv[27]) begin
    always @(posedge btn_clk) begin
        num <= num + 4;
    end
    
    assign leds[7] = btn_clk;
	/* rst always true */
	assign leds[6] = rst;
	assign leds[5] = SEGDT;
	assign leds[4] = clk200m;
	assign leds[3:0] = num[3:0];

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
        /* is rst remain  1 make it failed ?*/
    	.rst(rst),
    	// .rst(1'b0),
    	.Start(clkdiv[16]),
    	// .Start(clkdiv[6]),
    	.PData(seg),

    	.s_clk(SEGCLK),
    	.s_clrn(SEGCLR),
    	.sout(SEGDT),
    	.EN(SEGEN)
    );

endmodule 
