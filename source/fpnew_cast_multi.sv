module fpnew_cast_multi #(
    // Floating-point format configuration
    parameter fpnew_pkg::fmt_logic_t  FpFmtConfig  = '1,
    // Integer format configuration
    parameter fpnew_pkg::ifmt_logic_t IntFmtConfig = '1,

    // Number of pipeline registers
    parameter int unsigned             NumPipeRegs = 0,
    // Pipeline configuration (BEFORE, AFTER, DISTRIBUTED, etc.)
    parameter fpnew_pkg::pipe_config_t PipeConfig  = fpnew_pkg::BEFORE,
    // Type for tagging operations
    parameter type                     TagType     = logic,
    // Type for auxiliary data
    parameter type                     AuxType     = logic,

    // Local parameter for maximum data width
    localparam int unsigned WIDTH = fpnew_pkg::maximum(
        fpnew_pkg::max_fp_width(FpFmtConfig), fpnew_pkg::max_int_width(IntFmtConfig)
    ),
    // Number of floating-point formats supported
    localparam int unsigned NUM_FORMATS = fpnew_pkg::NUM_FP_FORMATS,
    // Width of the enable signal for pipeline registers
    localparam int unsigned ExtRegEnaWidth = NumPipeRegs == 0 ? 1 : NumPipeRegs
) (
    // Clock signal
    input logic clk_i,
    // Active-low reset signal
    input logic rst_ni,

    // Input operand
    input logic                   [      WIDTH-1:0] operands_i,
    // Indicates if the operand is boxed
    input logic                   [NUM_FORMATS-1:0] is_boxed_i,
    // Rounding mode for the operation
    input fpnew_pkg::roundmode_e                    rnd_mode_i,
    // Operation type (e.g., F2I, I2F)
    input fpnew_pkg::operation_e                    op_i,
    // Modifier for the operation
    input logic                                     op_mod_i,
    // Source floating-point format
    input fpnew_pkg::fp_format_e                    src_fmt_i,
    // Destination floating-point format
    input fpnew_pkg::fp_format_e                    dst_fmt_i,
    // Integer format for the operation
    input fpnew_pkg::int_format_e                   int_fmt_i,
    // Tag associated with the operation
    input TagType                                   tag_i,
    // Mask signal for the operation
    input logic                                     mask_i,
    // Auxiliary data for the operation
    input AuxType                                   aux_i,

    // Input valid signal
    input  logic in_valid_i,
    // Output ready signal for input stage
    output logic in_ready_o,
    // Flush signal to clear the pipeline
    input  logic flush_i,

    // Result of the operation
    output logic               [WIDTH-1:0] result_o,
    // Status of the operation
    output fpnew_pkg::status_t             status_o,
    // Extension bit for additional information
    output logic                           extension_bit_o,
    // Tag associated with the result
    output TagType                         tag_o,
    // Mask signal for the result
    output logic                           mask_o,
    // Auxiliary data for the result
    output AuxType                         aux_o,

    // Output valid signal
    output logic out_valid_o,
    // Input ready signal for output stage
    input  logic out_ready_i,

    // Busy signal indicating ongoing operation
    output logic busy_o,

    // Enable signals for pipeline registers
    input logic [ExtRegEnaWidth-1:0] reg_ena_i
);

  // Local parameters for integer formats and super format
  localparam int unsigned NUM_INT_FORMATS = fpnew_pkg::NUM_INT_FORMATS;
  localparam int unsigned MAX_INT_WIDTH = fpnew_pkg::max_int_width(IntFmtConfig);
  localparam fpnew_pkg::fp_encoding_t SUPER_FORMAT = fpnew_pkg::super_format(FpFmtConfig);

  // Local parameters for super format properties
  localparam int unsigned SUPER_EXP_BITS = SUPER_FORMAT.exp_bits;
  localparam int unsigned SUPER_MAN_BITS = SUPER_FORMAT.man_bits;
  localparam int unsigned SUPER_BIAS = 2 ** (SUPER_EXP_BITS - 1) - 1;

  // Local parameters for intermediate widths
  localparam int unsigned INT_MAN_WIDTH = fpnew_pkg::maximum(SUPER_MAN_BITS + 1, MAX_INT_WIDTH);
  localparam int unsigned LZC_RESULT_WIDTH = $clog2(INT_MAN_WIDTH);

  // Local parameters for integer exponent width
  localparam int unsigned INT_EXP_WIDTH = fpnew_pkg::maximum(
      $clog2(MAX_INT_WIDTH), fpnew_pkg::maximum(SUPER_EXP_BITS, $clog2(SUPER_BIAS + SUPER_MAN_BITS))
  ) + 1;

  // Number of pipeline registers for input, middle, and output stages
  localparam NUM_INP_REGS = PipeConfig == fpnew_pkg::BEFORE
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? ((NumPipeRegs + 1) / 3)
                               : 0);
  localparam NUM_MID_REGS = PipeConfig == fpnew_pkg::INSIDE
                          ? NumPipeRegs
                          : (PipeConfig == fpnew_pkg::DISTRIBUTED
                             ? ((NumPipeRegs + 2) / 3)
                             : 0);
  localparam NUM_OUT_REGS = PipeConfig == fpnew_pkg::AFTER
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? (NumPipeRegs / 3)
                               : 0);

  // Input pipeline registers for operands, formats, and control signals
  logic                   [      WIDTH-1:0]                  operands_q;
  logic                   [NUM_FORMATS-1:0]                  is_boxed_q;
  logic                                                      op_mod_q;
  fpnew_pkg::fp_format_e                                     src_fmt_q;
  fpnew_pkg::fp_format_e                                     dst_fmt_q;
  fpnew_pkg::int_format_e                                    int_fmt_q;

  logic                   [ 0:NUM_INP_REGS][      WIDTH-1:0] inp_pipe_operands_q;
  logic                   [ 0:NUM_INP_REGS][NUM_FORMATS-1:0] inp_pipe_is_boxed_q;
  fpnew_pkg::roundmode_e  [ 0:NUM_INP_REGS]                  inp_pipe_rnd_mode_q;
  fpnew_pkg::operation_e  [ 0:NUM_INP_REGS]                  inp_pipe_op_q;
  logic                   [ 0:NUM_INP_REGS]                  inp_pipe_op_mod_q;
  fpnew_pkg::fp_format_e  [ 0:NUM_INP_REGS]                  inp_pipe_src_fmt_q;
  fpnew_pkg::fp_format_e  [ 0:NUM_INP_REGS]                  inp_pipe_dst_fmt_q;
  fpnew_pkg::int_format_e [ 0:NUM_INP_REGS]                  inp_pipe_int_fmt_q;
  TagType                 [ 0:NUM_INP_REGS]                  inp_pipe_tag_q;
  logic                   [ 0:NUM_INP_REGS]                  inp_pipe_mask_q;
  AuxType                 [ 0:NUM_INP_REGS]                  inp_pipe_aux_q;
  logic                   [ 0:NUM_INP_REGS]                  inp_pipe_valid_q;

  logic                   [ 0:NUM_INP_REGS]                  inp_pipe_ready;

  assign inp_pipe_operands_q[0] = operands_i;
  assign inp_pipe_is_boxed_q[0] = is_boxed_i;
  assign inp_pipe_rnd_mode_q[0] = rnd_mode_i;
  assign inp_pipe_op_q[0]       = op_i;
  assign inp_pipe_op_mod_q[0]   = op_mod_i;
  assign inp_pipe_src_fmt_q[0]  = src_fmt_i;
  assign inp_pipe_dst_fmt_q[0]  = dst_fmt_i;
  assign inp_pipe_int_fmt_q[0]  = int_fmt_i;
  assign inp_pipe_tag_q[0]      = tag_i;
  assign inp_pipe_mask_q[0]     = mask_i;
  assign inp_pipe_aux_q[0]      = aux_i;
  assign inp_pipe_valid_q[0]    = in_valid_i;

  assign in_ready_o             = inp_pipe_ready[0];

  // Generate input pipeline stages
  for (genvar i = 0; i < NUM_INP_REGS; i++) begin : gen_input_pipeline
    logic reg_ena;

    assign inp_pipe_ready[i] = inp_pipe_ready[i+1] | ~inp_pipe_valid_q[i+1];

    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        inp_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        inp_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (inp_pipe_ready[i]) ? (inp_pipe_valid_q[i]) : (inp_pipe_valid_q[i+1]);
      end
    end

    assign reg_ena = (inp_pipe_ready[i] & inp_pipe_valid_q[i]) | reg_ena_i[i];

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_operands_q[i+1] <= ('0);
      end else begin
        inp_pipe_operands_q[i+1] <= (reg_ena) ? (inp_pipe_operands_q[i]) : (inp_pipe_operands_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_is_boxed_q[i+1] <= ('0);
      end else begin
        inp_pipe_is_boxed_q[i+1] <= (reg_ena) ? (inp_pipe_is_boxed_q[i]) : (inp_pipe_is_boxed_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_rnd_mode_q[i+1] <= (fpnew_pkg::RNE);
      end else begin
        inp_pipe_rnd_mode_q[i+1] <= (reg_ena) ? (inp_pipe_rnd_mode_q[i]) : (inp_pipe_rnd_mode_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_op_q[i+1] <= (fpnew_pkg::FMADD);
      end else begin
        inp_pipe_op_q[i+1] <= (reg_ena) ? (inp_pipe_op_q[i]) : (inp_pipe_op_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_op_mod_q[i+1] <= ('0);
      end else begin
        inp_pipe_op_mod_q[i+1] <= (reg_ena) ? (inp_pipe_op_mod_q[i]) : (inp_pipe_op_mod_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_src_fmt_q[i+1] <= (fpnew_pkg::fp_format_e'(0));
      end else begin
        inp_pipe_src_fmt_q[i+1] <= (reg_ena) ? (inp_pipe_src_fmt_q[i]) : (inp_pipe_src_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_dst_fmt_q[i+1] <= (fpnew_pkg::fp_format_e'(0));
      end else begin
        inp_pipe_dst_fmt_q[i+1] <= (reg_ena) ? (inp_pipe_dst_fmt_q[i]) : (inp_pipe_dst_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_int_fmt_q[i+1] <= (fpnew_pkg::int_format_e'(0));
      end else begin
        inp_pipe_int_fmt_q[i+1] <= (reg_ena) ? (inp_pipe_int_fmt_q[i]) : (inp_pipe_int_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_tag_q[i+1] <= (TagType'('0));
      end else begin
        inp_pipe_tag_q[i+1] <= (reg_ena) ? (inp_pipe_tag_q[i]) : (inp_pipe_tag_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_mask_q[i+1] <= ('0);
      end else begin
        inp_pipe_mask_q[i+1] <= (reg_ena) ? (inp_pipe_mask_q[i]) : (inp_pipe_mask_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        inp_pipe_aux_q[i+1] <= (AuxType'('0));
      end else begin
        inp_pipe_aux_q[i+1] <= (reg_ena) ? (inp_pipe_aux_q[i]) : (inp_pipe_aux_q[i+1]);
      end
    end

  end

  assign operands_q = inp_pipe_operands_q[NUM_INP_REGS];
  assign is_boxed_q = inp_pipe_is_boxed_q[NUM_INP_REGS];
  assign op_mod_q   = inp_pipe_op_mod_q[NUM_INP_REGS];
  assign src_fmt_q  = inp_pipe_src_fmt_q[NUM_INP_REGS];
  assign dst_fmt_q  = inp_pipe_dst_fmt_q[NUM_INP_REGS];
  assign int_fmt_q  = inp_pipe_int_fmt_q[NUM_INP_REGS];

  logic src_is_int, dst_is_int;

  assign src_is_int = (inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::I2F);
  assign dst_is_int = (inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::F2I);

  logic                [  INT_MAN_WIDTH-1:0]                    encoded_mant;

  logic                [    NUM_FORMATS-1:0]                    fmt_sign;
  logic signed         [    NUM_FORMATS-1:0][INT_EXP_WIDTH-1:0] fmt_exponent;
  logic                [    NUM_FORMATS-1:0][INT_MAN_WIDTH-1:0] fmt_mantissa;
  logic signed         [    NUM_FORMATS-1:0][INT_EXP_WIDTH-1:0] fmt_shift_compensation;

  fpnew_pkg::fp_info_t [    NUM_FORMATS-1:0]                    info;

  logic                [NUM_INT_FORMATS-1:0][INT_MAN_WIDTH-1:0] ifmt_input_val;
  logic                                                         int_sign;
  logic [INT_MAN_WIDTH-1:0] int_value, int_mantissa;

  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : fmt_init_inputs
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    if (FpFmtConfig[fmt]) begin : active_format
      fpnew_classifier #(
          .FpFormat   (fpnew_pkg::fp_format_e'(fmt)),
          .NumOperands(1)
      ) i_fpnew_classifier (
          .operands_i(operands_q[FP_WIDTH-1:0]),
          .is_boxed_i(is_boxed_q[fmt]),
          .info_o    (info[fmt])
      );

      assign fmt_sign[fmt]               = operands_q[FP_WIDTH-1];
      assign fmt_exponent[fmt]           = signed'({1'b0, operands_q[MAN_BITS+:EXP_BITS]});
      assign fmt_mantissa[fmt]           = {info[fmt].is_normal, operands_q[MAN_BITS-1:0]};

      assign fmt_shift_compensation[fmt] = signed'(INT_MAN_WIDTH - 1 - MAN_BITS);
    end else begin : inactive_format
      assign info[fmt]                   = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_sign[fmt]               = fpnew_pkg::DONT_CARE;
      assign fmt_exponent[fmt]           = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_mantissa[fmt]           = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_shift_compensation[fmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  for (genvar ifmt = 0; ifmt < int'(NUM_INT_FORMATS); ifmt++) begin : gen_sign_extend_int
    localparam int unsigned INT_WIDTH = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));

    if (IntFmtConfig[ifmt]) begin : active_format
      always_comb begin : sign_ext_input
        ifmt_input_val[ifmt]                = '{default: operands_q[INT_WIDTH-1] & ~op_mod_q};
        ifmt_input_val[ifmt][INT_WIDTH-1:0] = operands_q[INT_WIDTH-1:0];
      end
    end else begin : inactive_format
      assign ifmt_input_val[ifmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  assign int_value    = ifmt_input_val[int_fmt_q];
  assign int_sign     = int_value[INT_MAN_WIDTH-1] & ~op_mod_q;
  assign int_mantissa = int_sign ? unsigned'(-int_value) : int_value;

  assign encoded_mant = src_is_int ? int_mantissa : fmt_mantissa[src_fmt_q];

  logic signed [INT_EXP_WIDTH-1:0] src_bias;
  logic signed [INT_EXP_WIDTH-1:0] src_exp;
  logic signed [INT_EXP_WIDTH-1:0] src_subnormal;
  logic signed [INT_EXP_WIDTH-1:0] src_offset;

  assign src_bias      = signed'(fpnew_pkg::bias(src_fmt_q));
  assign src_exp       = fmt_exponent[src_fmt_q];
  assign src_subnormal = signed'({1'b0, info[src_fmt_q].is_subnormal});
  assign src_offset    = fmt_shift_compensation[src_fmt_q];

  logic                               input_sign;
  logic signed [   INT_EXP_WIDTH-1:0] input_exp;
  logic        [   INT_MAN_WIDTH-1:0] input_mant;
  logic                               mant_is_zero;

  logic signed [   INT_EXP_WIDTH-1:0] fp_input_exp;
  logic signed [   INT_EXP_WIDTH-1:0] int_input_exp;

  logic        [LZC_RESULT_WIDTH-1:0] renorm_shamt;
  logic        [  LZC_RESULT_WIDTH:0] renorm_shamt_sgn;

  lzc #(
      .WIDTH(INT_MAN_WIDTH),
      .MODE (1)
  ) i_lzc (
      .in_i   (encoded_mant),
      .cnt_o  (renorm_shamt),
      .empty_o(mant_is_zero)
  );
  assign renorm_shamt_sgn = signed'({1'b0, renorm_shamt});

  assign input_sign = src_is_int ? int_sign : fmt_sign[src_fmt_q];

  assign input_mant = encoded_mant << renorm_shamt;

  assign fp_input_exp = signed'(src_exp + src_subnormal - src_bias - renorm_shamt_sgn + src_offset);
  assign int_input_exp = signed'(INT_MAN_WIDTH - 1 - renorm_shamt_sgn);

  assign input_exp = src_is_int ? int_input_exp : fp_input_exp;

  logic signed [INT_EXP_WIDTH-1:0] destination_exp;

  assign destination_exp = input_exp + signed'(fpnew_pkg::bias(dst_fmt_q));

  logic                                                          input_sign_q;
  logic signed            [INT_EXP_WIDTH-1:0]                    input_exp_q;
  logic                   [INT_MAN_WIDTH-1:0]                    input_mant_q;
  logic signed            [INT_EXP_WIDTH-1:0]                    destination_exp_q;
  logic                                                          src_is_int_q;
  logic                                                          dst_is_int_q;
  fpnew_pkg::fp_info_t                                           info_q;
  logic                                                          mant_is_zero_q;
  logic                                                          op_mod_q2;
  fpnew_pkg::roundmode_e                                         rnd_mode_q;
  fpnew_pkg::fp_format_e                                         src_fmt_q2;
  fpnew_pkg::fp_format_e                                         dst_fmt_q2;
  fpnew_pkg::int_format_e                                        int_fmt_q2;

  logic                   [   0:NUM_MID_REGS]                    mid_pipe_input_sign_q;
  logic signed            [   0:NUM_MID_REGS][INT_EXP_WIDTH-1:0] mid_pipe_input_exp_q;
  logic                   [   0:NUM_MID_REGS][INT_MAN_WIDTH-1:0] mid_pipe_input_mant_q;
  logic signed            [   0:NUM_MID_REGS][INT_EXP_WIDTH-1:0] mid_pipe_dest_exp_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_src_is_int_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_dst_is_int_q;
  fpnew_pkg::fp_info_t    [   0:NUM_MID_REGS]                    mid_pipe_info_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_mant_zero_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_op_mod_q;
  fpnew_pkg::roundmode_e  [   0:NUM_MID_REGS]                    mid_pipe_rnd_mode_q;
  fpnew_pkg::fp_format_e  [   0:NUM_MID_REGS]                    mid_pipe_src_fmt_q;
  fpnew_pkg::fp_format_e  [   0:NUM_MID_REGS]                    mid_pipe_dst_fmt_q;
  fpnew_pkg::int_format_e [   0:NUM_MID_REGS]                    mid_pipe_int_fmt_q;
  TagType                 [   0:NUM_MID_REGS]                    mid_pipe_tag_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_mask_q;
  AuxType                 [   0:NUM_MID_REGS]                    mid_pipe_aux_q;
  logic                   [   0:NUM_MID_REGS]                    mid_pipe_valid_q;

  logic                   [   0:NUM_MID_REGS]                    mid_pipe_ready;

  assign mid_pipe_input_sign_q[0]     = input_sign;
  assign mid_pipe_input_exp_q[0]      = input_exp;
  assign mid_pipe_input_mant_q[0]     = input_mant;
  assign mid_pipe_dest_exp_q[0]       = destination_exp;
  assign mid_pipe_src_is_int_q[0]     = src_is_int;
  assign mid_pipe_dst_is_int_q[0]     = dst_is_int;
  assign mid_pipe_info_q[0]           = info[src_fmt_q];
  assign mid_pipe_mant_zero_q[0]      = mant_is_zero;
  assign mid_pipe_op_mod_q[0]         = op_mod_q;
  assign mid_pipe_rnd_mode_q[0]       = inp_pipe_rnd_mode_q[NUM_INP_REGS];
  assign mid_pipe_src_fmt_q[0]        = src_fmt_q;
  assign mid_pipe_dst_fmt_q[0]        = dst_fmt_q;
  assign mid_pipe_int_fmt_q[0]        = int_fmt_q;
  assign mid_pipe_tag_q[0]            = inp_pipe_tag_q[NUM_INP_REGS];
  assign mid_pipe_mask_q[0]           = inp_pipe_mask_q[NUM_INP_REGS];
  assign mid_pipe_aux_q[0]            = inp_pipe_aux_q[NUM_INP_REGS];
  assign mid_pipe_valid_q[0]          = inp_pipe_valid_q[NUM_INP_REGS];

  assign inp_pipe_ready[NUM_INP_REGS] = mid_pipe_ready[0];

  // Generate middle pipeline stages
  for (genvar i = 0; i < NUM_MID_REGS; i++) begin : gen_inside_pipeline
    logic reg_ena;

    assign mid_pipe_ready[i] = mid_pipe_ready[i+1] | ~mid_pipe_valid_q[i+1];

    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        mid_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        mid_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (mid_pipe_ready[i]) ? (mid_pipe_valid_q[i]) : (mid_pipe_valid_q[i+1]);
      end
    end

    assign reg_ena = (mid_pipe_ready[i] & mid_pipe_valid_q[i]) | reg_ena_i[NUM_INP_REGS+i];

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_input_sign_q[i+1] <= ('0);
      end else begin
        mid_pipe_input_sign_q[i+1] <= (reg_ena) ? (mid_pipe_input_sign_q[i]) : (mid_pipe_input_sign_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_input_exp_q[i+1] <= ('0);
      end else begin
        mid_pipe_input_exp_q[i+1] <= (reg_ena) ? (mid_pipe_input_exp_q[i]) : (mid_pipe_input_exp_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_input_mant_q[i+1] <= ('0);
      end else begin
        mid_pipe_input_mant_q[i+1] <= (reg_ena) ? (mid_pipe_input_mant_q[i]) : (mid_pipe_input_mant_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_dest_exp_q[i+1] <= ('0);
      end else begin
        mid_pipe_dest_exp_q[i+1] <= (reg_ena) ? (mid_pipe_dest_exp_q[i]) : (mid_pipe_dest_exp_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_src_is_int_q[i+1] <= ('0);
      end else begin
        mid_pipe_src_is_int_q[i+1] <= (reg_ena) ? (mid_pipe_src_is_int_q[i]) : (mid_pipe_src_is_int_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_dst_is_int_q[i+1] <= ('0);
      end else begin
        mid_pipe_dst_is_int_q[i+1] <= (reg_ena) ? (mid_pipe_dst_is_int_q[i]) : (mid_pipe_dst_is_int_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_info_q[i+1] <= ('0);
      end else begin
        mid_pipe_info_q[i+1] <= (reg_ena) ? (mid_pipe_info_q[i]) : (mid_pipe_info_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_mant_zero_q[i+1] <= ('0);
      end else begin
        mid_pipe_mant_zero_q[i+1] <= (reg_ena) ? (mid_pipe_mant_zero_q[i]) : (mid_pipe_mant_zero_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_op_mod_q[i+1] <= ('0);
      end else begin
        mid_pipe_op_mod_q[i+1] <= (reg_ena) ? (mid_pipe_op_mod_q[i]) : (mid_pipe_op_mod_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_rnd_mode_q[i+1] <= (fpnew_pkg::RNE);
      end else begin
        mid_pipe_rnd_mode_q[i+1] <= (reg_ena) ? (mid_pipe_rnd_mode_q[i]) : (mid_pipe_rnd_mode_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_src_fmt_q[i+1] <= (fpnew_pkg::fp_format_e'(0));
      end else begin
        mid_pipe_src_fmt_q[i+1] <= (reg_ena) ? (mid_pipe_src_fmt_q[i]) : (mid_pipe_src_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_dst_fmt_q[i+1] <= (fpnew_pkg::fp_format_e'(0));
      end else begin
        mid_pipe_dst_fmt_q[i+1] <= (reg_ena) ? (mid_pipe_dst_fmt_q[i]) : (mid_pipe_dst_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_int_fmt_q[i+1] <= (fpnew_pkg::int_format_e'(0));
      end else begin
        mid_pipe_int_fmt_q[i+1] <= (reg_ena) ? (mid_pipe_int_fmt_q[i]) : (mid_pipe_int_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_tag_q[i+1] <= (TagType'('0));
      end else begin
        mid_pipe_tag_q[i+1] <= (reg_ena) ? (mid_pipe_tag_q[i]) : (mid_pipe_tag_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_mask_q[i+1] <= ('0);
      end else begin
        mid_pipe_mask_q[i+1] <= (reg_ena) ? (mid_pipe_mask_q[i]) : (mid_pipe_mask_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_aux_q[i+1] <= (AuxType'('0));
      end else begin
        mid_pipe_aux_q[i+1] <= (reg_ena) ? (mid_pipe_aux_q[i]) : (mid_pipe_aux_q[i+1]);
      end
    end

  end

  assign input_sign_q      = mid_pipe_input_sign_q[NUM_MID_REGS];
  assign input_exp_q       = mid_pipe_input_exp_q[NUM_MID_REGS];
  assign input_mant_q      = mid_pipe_input_mant_q[NUM_MID_REGS];
  assign destination_exp_q = mid_pipe_dest_exp_q[NUM_MID_REGS];
  assign src_is_int_q      = mid_pipe_src_is_int_q[NUM_MID_REGS];
  assign dst_is_int_q      = mid_pipe_dst_is_int_q[NUM_MID_REGS];
  assign info_q            = mid_pipe_info_q[NUM_MID_REGS];
  assign mant_is_zero_q    = mid_pipe_mant_zero_q[NUM_MID_REGS];
  assign op_mod_q2         = mid_pipe_op_mod_q[NUM_MID_REGS];
  assign rnd_mode_q        = mid_pipe_rnd_mode_q[NUM_MID_REGS];
  assign src_fmt_q2        = mid_pipe_src_fmt_q[NUM_MID_REGS];
  assign dst_fmt_q2        = mid_pipe_dst_fmt_q[NUM_MID_REGS];
  assign int_fmt_q2        = mid_pipe_int_fmt_q[NUM_MID_REGS];

  logic [INT_EXP_WIDTH-1:0] final_exp;

  logic [2*INT_MAN_WIDTH:0] preshift_mant;
  logic [2*INT_MAN_WIDTH:0] destination_mant;
  logic [SUPER_MAN_BITS-1:0] final_mant;
  logic [MAX_INT_WIDTH-1:0] final_int;

  logic [$clog2(INT_MAN_WIDTH+1)-1:0] denorm_shamt;

  logic [1:0] fp_round_sticky_bits, int_round_sticky_bits, round_sticky_bits;
  logic of_before_round, uf_before_round;

  always_comb begin : cast_value
    final_exp       = unsigned'(destination_exp_q);
    preshift_mant   = '0;
    denorm_shamt    = SUPER_MAN_BITS - fpnew_pkg::man_bits(dst_fmt_q2);
    of_before_round = 1'b0;
    uf_before_round = 1'b0;

    preshift_mant   = input_mant_q << (INT_MAN_WIDTH + 1);

    if (dst_is_int_q) begin
      denorm_shamt = unsigned'(MAX_INT_WIDTH - 1 - input_exp_q);

      if ((input_exp_q >= signed'(fpnew_pkg::int_width(
              int_fmt_q2
          ) - 1 + op_mod_q2)) &&
              !(!op_mod_q2 && input_sign_q && (input_exp_q == signed'(fpnew_pkg::int_width(
              int_fmt_q2
          ) - 1)) && (input_mant_q == {1'b1, {INT_MAN_WIDTH - 1{1'b0}}}))) begin
        denorm_shamt    = '0;
        of_before_round = 1'b1;
      end else if (input_exp_q < -1) begin
        denorm_shamt    = MAX_INT_WIDTH + 1;
        uf_before_round = 1'b1;
      end
    end else begin
      if ((destination_exp_q >= signed'(2 ** fpnew_pkg::exp_bits(
              dst_fmt_q2
          )) - 1) || (~src_is_int_q && info_q.is_inf)) begin
        final_exp       = unsigned'(2 ** fpnew_pkg::exp_bits(dst_fmt_q2) - 2);
        preshift_mant   = '1;
        of_before_round = 1'b1;
      end else if (destination_exp_q < 1 && destination_exp_q >= -signed'(fpnew_pkg::man_bits(
              dst_fmt_q2
          ))) begin
        final_exp       = '0;
        denorm_shamt    = unsigned'(denorm_shamt + 1 - destination_exp_q);
        uf_before_round = 1'b1;
      end else if (destination_exp_q < -signed'(fpnew_pkg::man_bits(dst_fmt_q2))) begin
        final_exp       = '0;
        denorm_shamt    = unsigned'(denorm_shamt + 2 + fpnew_pkg::man_bits(dst_fmt_q2));
        uf_before_round = 1'b1;
      end
    end
  end

  localparam NUM_FP_STICKY = 2 * INT_MAN_WIDTH - SUPER_MAN_BITS - 1;
  localparam NUM_INT_STICKY = 2 * INT_MAN_WIDTH - MAX_INT_WIDTH;

  assign destination_mant = preshift_mant >> denorm_shamt;

  assign {final_mant, fp_round_sticky_bits[1]} =
      destination_mant[2*INT_MAN_WIDTH-1-:SUPER_MAN_BITS+1];
  assign {final_int, int_round_sticky_bits[1]} = destination_mant[2*INT_MAN_WIDTH-:MAX_INT_WIDTH+1];

  assign fp_round_sticky_bits[0] = (|{destination_mant[NUM_FP_STICKY-1:0]});
  assign int_round_sticky_bits[0] = (|{destination_mant[NUM_INT_STICKY-1:0]});

  assign round_sticky_bits = dst_is_int_q ? int_round_sticky_bits : fp_round_sticky_bits;

  logic [          WIDTH-1:0]            pre_round_abs;
  logic                                  of_after_round;
  logic                                  uf_after_round;

  logic [    NUM_FORMATS-1:0][WIDTH-1:0] fmt_pre_round_abs;
  logic [    NUM_FORMATS-1:0]            fmt_of_after_round;
  logic [    NUM_FORMATS-1:0]            fmt_uf_after_round;

  logic [NUM_INT_FORMATS-1:0][WIDTH-1:0] ifmt_pre_round_abs;
  logic [NUM_INT_FORMATS-1:0]            ifmt_of_after_round;

  logic                                  rounded_sign;
  logic [          WIDTH-1:0]            rounded_abs;
  logic                                  result_true_zero;

  logic [          WIDTH-1:0]            rounded_int_res;
  logic                                  rounded_int_res_zero;

  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_res_assemble
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    if (FpFmtConfig[fmt]) begin : active_format
      always_comb begin : assemble_result
        fmt_pre_round_abs[fmt] = {final_exp[EXP_BITS-1:0], final_mant[MAN_BITS-1:0]};
      end
    end else begin : inactive_format
      assign fmt_pre_round_abs[fmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  for (genvar ifmt = 0; ifmt < int'(NUM_INT_FORMATS); ifmt++) begin : gen_int_res_sign_ext
    localparam int unsigned INT_WIDTH = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));

    if (IntFmtConfig[ifmt]) begin : active_format
      always_comb begin : assemble_result
        ifmt_pre_round_abs[ifmt]                = '{default: final_int[INT_WIDTH-1]};
        ifmt_pre_round_abs[ifmt][INT_WIDTH-1:0] = final_int[INT_WIDTH-1:0];
      end
    end else begin : inactive_format
      assign ifmt_pre_round_abs[ifmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  assign pre_round_abs = dst_is_int_q ? ifmt_pre_round_abs[int_fmt_q2] : fmt_pre_round_abs[dst_fmt_q2];

  fpnew_rounding #(
      .AbsWidth(WIDTH)
  ) i_fpnew_rounding (
      .abs_value_i            (pre_round_abs),
      .sign_i                 (input_sign_q),
      .round_sticky_bits_i    (round_sticky_bits),
      .rnd_mode_i             (rnd_mode_q),
      .effective_subtraction_i(1'b0),
      .abs_rounded_o          (rounded_abs),
      .sign_o                 (rounded_sign),
      .exact_zero_o           (result_true_zero)
  );

  logic [NUM_FORMATS-1:0][WIDTH-1:0] fmt_result;

  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_sign_inject
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    if (FpFmtConfig[fmt]) begin : active_format
      always_comb begin : post_process
        fmt_uf_after_round[fmt] = rounded_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == '0;
        fmt_of_after_round[fmt] = rounded_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == '1;

        fmt_result[fmt] = '1;
        fmt_result[fmt][FP_WIDTH-1:0] = src_is_int_q & mant_is_zero_q
                                        ? '0
                                        : {rounded_sign, rounded_abs[EXP_BITS+MAN_BITS-1:0]};
      end
    end else begin : inactive_format
      assign fmt_uf_after_round[fmt] = fpnew_pkg::DONT_CARE;
      assign fmt_of_after_round[fmt] = fpnew_pkg::DONT_CARE;
      assign fmt_result[fmt]         = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  assign rounded_int_res      = rounded_sign ? unsigned'(-rounded_abs) : rounded_abs;
  assign rounded_int_res_zero = (rounded_int_res == '0);

  for (genvar ifmt = 0; ifmt < int'(NUM_INT_FORMATS); ifmt++) begin : gen_int_overflow
    localparam int unsigned INT_WIDTH = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));

    if (IntFmtConfig[ifmt]) begin : active_format
      always_comb begin : detect_overflow
        ifmt_of_after_round[ifmt] = 1'b0;

        if (!rounded_sign && input_exp_q == signed'(INT_WIDTH - 2 + op_mod_q2)) begin
          ifmt_of_after_round[ifmt] = ~rounded_int_res[INT_WIDTH-2+op_mod_q2];
        end
      end
    end else begin : inactive_format
      assign ifmt_of_after_round[ifmt] = fpnew_pkg::DONT_CARE;
    end
  end

  assign uf_after_round = fmt_uf_after_round[dst_fmt_q2];
  assign of_after_round = dst_is_int_q ? ifmt_of_after_round[int_fmt_q2] : fmt_of_after_round[dst_fmt_q2];

  logic               [      WIDTH-1:0]            fp_special_result;
  fpnew_pkg::status_t                              fp_special_status;
  logic                                            fp_result_is_special;

  logic               [NUM_FORMATS-1:0][WIDTH-1:0] fmt_special_result;

  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_special_results
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    localparam logic [EXP_BITS-1:0] QNAN_EXPONENT = '1;
    localparam logic [MAN_BITS-1:0] QNAN_MANTISSA = 2 ** (MAN_BITS - 1);

    if (FpFmtConfig[fmt]) begin : active_format
      always_comb begin : special_results
        logic [FP_WIDTH-1:0] special_res;
        special_res = info_q.is_zero
                      ? input_sign_q << FP_WIDTH-1
                      : {1'b0, QNAN_EXPONENT, QNAN_MANTISSA};

        fmt_special_result[fmt] = '1;
        fmt_special_result[fmt][FP_WIDTH-1:0] = special_res;
      end
    end else begin : inactive_format
      assign fmt_special_result[fmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  assign fp_result_is_special = ~src_is_int_q & (info_q.is_zero | info_q.is_nan | ~info_q.is_boxed);

  assign fp_special_status = '{NV: info_q.is_signalling, default: 1'b0};

  assign fp_special_result = fmt_special_result[dst_fmt_q2];

  logic               [          WIDTH-1:0]            int_special_result;
  fpnew_pkg::status_t                                  int_special_status;
  logic                                                int_result_is_special;

  logic               [NUM_INT_FORMATS-1:0][WIDTH-1:0] ifmt_special_result;

  for (genvar ifmt = 0; ifmt < int'(NUM_INT_FORMATS); ifmt++) begin : gen_special_results_int
    localparam int unsigned INT_WIDTH = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));

    if (IntFmtConfig[ifmt]) begin : active_format
      always_comb begin : special_results
        automatic logic [INT_WIDTH-1:0] special_res;

        special_res[INT_WIDTH-2:0] = '1;
        special_res[INT_WIDTH-1]   = op_mod_q2;

        if (input_sign_q && !info_q.is_nan) special_res = ~special_res;

        ifmt_special_result[ifmt]                = '{default: special_res[INT_WIDTH-1]};
        ifmt_special_result[ifmt][INT_WIDTH-1:0] = special_res;
      end
    end else begin : inactive_format
      assign ifmt_special_result[ifmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  assign int_result_is_special = info_q.is_nan | info_q.is_inf |
                                 of_before_round | of_after_round | ~info_q.is_boxed |
                                 (input_sign_q & op_mod_q2 & ~rounded_int_res_zero);

  assign int_special_status = '{NV: 1'b1, default: 1'b0};

  assign int_special_result = ifmt_special_result[int_fmt_q2];

  fpnew_pkg::status_t int_regular_status, fp_regular_status;

  logic [WIDTH-1:0] fp_result, int_result;
  fpnew_pkg::status_t fp_status, int_status;

  assign fp_regular_status.NV = src_is_int_q & (of_before_round | of_after_round);
  assign fp_regular_status.DZ = 1'b0;
  assign fp_regular_status.OF = ~src_is_int_q & (~info_q.is_inf & (of_before_round | of_after_round));
  assign fp_regular_status.UF = uf_after_round & fp_regular_status.NX;
  assign fp_regular_status.NX = src_is_int_q ? (| fp_round_sticky_bits)
            : (| fp_round_sticky_bits) | (~info_q.is_inf & (of_before_round | of_after_round));
  assign int_regular_status = '{NX: (|int_round_sticky_bits), default: 1'b0};

  assign fp_result = fp_result_is_special ? fp_special_result : fmt_result[dst_fmt_q2];
  assign fp_status = fp_result_is_special ? fp_special_status : fp_regular_status;
  assign int_result = int_result_is_special ? int_special_result : rounded_int_res;
  assign int_status = int_result_is_special ? int_special_status : int_regular_status;

  logic               [WIDTH-1:0] result_d;
  fpnew_pkg::status_t             status_d;
  logic                           extension_bit;

  assign result_d = dst_is_int_q ? int_result : fp_result;
  assign status_d = dst_is_int_q ? int_status : fp_status;

  assign extension_bit = dst_is_int_q ? int_result[WIDTH-1] : 1'b1;

  logic               [0:NUM_OUT_REGS][WIDTH-1:0] out_pipe_result_q;
  fpnew_pkg::status_t [0:NUM_OUT_REGS]            out_pipe_status_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_ext_bit_q;
  TagType             [0:NUM_OUT_REGS]            out_pipe_tag_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_mask_q;
  AuxType             [0:NUM_OUT_REGS]            out_pipe_aux_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_valid_q;

  logic               [0:NUM_OUT_REGS]            out_pipe_ready;

  assign out_pipe_result_q[0]         = result_d;
  assign out_pipe_status_q[0]         = status_d;
  assign out_pipe_ext_bit_q[0]        = extension_bit;
  assign out_pipe_tag_q[0]            = mid_pipe_tag_q[NUM_MID_REGS];
  assign out_pipe_mask_q[0]           = mid_pipe_mask_q[NUM_MID_REGS];
  assign out_pipe_aux_q[0]            = mid_pipe_aux_q[NUM_MID_REGS];
  assign out_pipe_valid_q[0]          = mid_pipe_valid_q[NUM_MID_REGS];

  assign mid_pipe_ready[NUM_MID_REGS] = out_pipe_ready[0];

  // Generate output pipeline stages
  for (genvar i = 0; i < NUM_OUT_REGS; i++) begin : gen_output_pipeline
    logic reg_ena;

    assign out_pipe_ready[i] = out_pipe_ready[i+1] | ~out_pipe_valid_q[i+1];

    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        out_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        out_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (out_pipe_ready[i]) ? (out_pipe_valid_q[i]) : (out_pipe_valid_q[i+1]);
      end
    end

    assign reg_ena = (out_pipe_ready[i] & out_pipe_valid_q[i]) | reg_ena_i[NUM_INP_REGS + NUM_MID_REGS + i];

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_result_q[i+1] <= ('0);
      end else begin
        out_pipe_result_q[i+1] <= (reg_ena) ? (out_pipe_result_q[i]) : (out_pipe_result_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_status_q[i+1] <= ('0);
      end else begin
        out_pipe_status_q[i+1] <= (reg_ena) ? (out_pipe_status_q[i]) : (out_pipe_status_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_ext_bit_q[i+1] <= ('0);
      end else begin
        out_pipe_ext_bit_q[i+1] <= (reg_ena) ? (out_pipe_ext_bit_q[i]) : (out_pipe_ext_bit_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_tag_q[i+1] <= (TagType'('0));
      end else begin
        out_pipe_tag_q[i+1] <= (reg_ena) ? (out_pipe_tag_q[i]) : (out_pipe_tag_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_mask_q[i+1] <= ('0);
      end else begin
        out_pipe_mask_q[i+1] <= (reg_ena) ? (out_pipe_mask_q[i]) : (out_pipe_mask_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_aux_q[i+1] <= (AuxType'('0));
      end else begin
        out_pipe_aux_q[i+1] <= (reg_ena) ? (out_pipe_aux_q[i]) : (out_pipe_aux_q[i+1]);
      end
    end

  end

  assign out_pipe_ready[NUM_OUT_REGS] = out_ready_i;

  assign result_o                     = out_pipe_result_q[NUM_OUT_REGS];
  assign status_o                     = out_pipe_status_q[NUM_OUT_REGS];
  assign extension_bit_o              = out_pipe_ext_bit_q[NUM_OUT_REGS];
  assign tag_o                        = out_pipe_tag_q[NUM_OUT_REGS];
  assign mask_o                       = out_pipe_mask_q[NUM_OUT_REGS];
  assign aux_o                        = out_pipe_aux_q[NUM_OUT_REGS];
  assign out_valid_o                  = out_pipe_valid_q[NUM_OUT_REGS];
  assign busy_o                       = (|{inp_pipe_valid_q, mid_pipe_valid_q, out_pipe_valid_q});
endmodule
