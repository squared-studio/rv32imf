// Module definition for a multi-format floating-point FMA unit.
module fpnew_fma_multi #(
    // Parameter for floating-point format configuration.
    parameter fpnew_pkg::fmt_logic_t   FpFmtConfig = '1,
    // Parameter for the number of pipeline registers.
    parameter int unsigned             NumPipeRegs = 0,
    // Parameter for pipeline configuration (BEFORE, INSIDE, AFTER, DISTRIBUTED).
    parameter fpnew_pkg::pipe_config_t PipeConfig  = fpnew_pkg::BEFORE,
    // Parameter for the data type of the tag signal.
    parameter type                     TagType     = logic,
    // Parameter for the data type of the auxiliary signal.
    parameter type                     AuxType     = logic,

    // Local parameter for the maximum floating-point width based on the format config.
    localparam int unsigned WIDTH          = fpnew_pkg::max_fp_width(FpFmtConfig),
    // Local parameter for the number of supported floating-point formats.
    localparam int unsigned NUM_FORMATS    = fpnew_pkg::NUM_FP_FORMATS,
    // Local parameter for the width of the register enable signal for external control.
    localparam int unsigned ExtRegEnaWidth = NumPipeRegs == 0 ? 1 : NumPipeRegs
) (
    // Input clock signal.
    input logic clk_i,
    // Input reset signal (active low).
    input logic rst_ni,

    // Input operands (three operands for FMA).
    input logic                  [            2:0][WIDTH-1:0] operands_i,
    // Input flags indicating if the operands are in boxed format.
    input logic                  [NUM_FORMATS-1:0][      2:0] is_boxed_i,
    // Input rounding mode.
    input fpnew_pkg::roundmode_e                              rnd_mode_i,
    // Input operation type (e.g., FMADD, FNMADD, ADD, MUL).
    input fpnew_pkg::operation_e                              op_i,
    // Input operation modifier (e.g., for negation).
    input logic                                               op_mod_i,
    // Input source operand format.
    input fpnew_pkg::fp_format_e                              src_fmt_i,
    // Input destination operand format.
    input fpnew_pkg::fp_format_e                              dst_fmt_i,
    // Input tag signal.
    input TagType                                             tag_i,
    // Input mask signal.
    input logic                                               mask_i,
    // Input auxiliary signal.
    input AuxType                                             aux_i,

    // Input validity signal for the input data.
    input  logic in_valid_i,
    // Output ready signal indicating the module can accept new input.
    output logic in_ready_o,
    // Input flush signal to reset the pipeline.
    input  logic flush_i,

    // Output result of the FMA operation.
    output logic               [WIDTH-1:0] result_o,
    // Output status flags (e.g., overflow, underflow, invalid operation).
    output fpnew_pkg::status_t             status_o,
    // Output extension bit (typically for custom extensions).
    output logic                           extension_bit_o,
    // Output tag signal corresponding to the result.
    output TagType                         tag_o,
    // Output mask signal corresponding to the result.
    output logic                           mask_o,
    // Output auxiliary signal corresponding to the result.
    output AuxType                         aux_o,

    // Output validity signal for the output data.
    output logic out_valid_o,
    // Input ready signal indicating the receiving module can accept the output.
    input  logic out_ready_i,

    // Output busy signal indicating the module is currently processing data.
    output logic busy_o,

    // Input register enable signal for pipeline registers.
    input logic [ExtRegEnaWidth-1:0] reg_ena_i
);


  // Local parameter for the super format (maximum width format).
  localparam fpnew_pkg::fp_encoding_t SUPER_FORMAT = fpnew_pkg::super_format(FpFmtConfig);

  // Local parameter for the number of exponent bits in the super format.
  localparam int unsigned SUPER_EXP_BITS = SUPER_FORMAT.exp_bits;
  // Local parameter for the number of mantissa bits in the super format.
  localparam int unsigned SUPER_MAN_BITS = SUPER_FORMAT.man_bits;


  // Local parameter for the total number of precision bits (mantissa + implicit one).
  localparam int unsigned PRECISION_BITS = SUPER_MAN_BITS + 1;

  // Local parameter for the width of the lower part of the sum.
  localparam int unsigned LOWER_SUM_WIDTH = 2 * PRECISION_BITS + 3;
  // Local parameter for the width of the leading zero counter result.
  localparam int unsigned LZC_RESULT_WIDTH = $clog2(LOWER_SUM_WIDTH);


  // Local parameter for the width of the exponent (including guard bits).
  localparam int unsigned EXP_WIDTH = fpnew_pkg::maximum(SUPER_EXP_BITS + 2, LZC_RESULT_WIDTH);

  // Local parameter for the width of the shift amount.
  localparam int unsigned SHIFT_AMOUNT_WIDTH = $clog2(3 * PRECISION_BITS + 5);

  // Local parameter for the number of input pipeline registers based on configuration.
  localparam NUM_INP_REGS = PipeConfig == fpnew_pkg::BEFORE
                              ? NumPipeRegs
                              : (PipeConfig == fpnew_pkg::DISTRIBUTED
                                 ? ((NumPipeRegs + 1) / 3)
                                 : 0);
  // Local parameter for the number of middle pipeline registers based on configuration.
  localparam NUM_MID_REGS = PipeConfig == fpnew_pkg::INSIDE
                              ? NumPipeRegs
                              : (PipeConfig == fpnew_pkg::DISTRIBUTED
                                 ? ((NumPipeRegs + 2) / 3)
                                 : 0);
  // Local parameter for the number of output pipeline registers based on configuration.
  localparam NUM_OUT_REGS = PipeConfig == fpnew_pkg::AFTER
                              ? NumPipeRegs
                              : (PipeConfig == fpnew_pkg::DISTRIBUTED
                                 ? (NumPipeRegs / 3)
                                 : 0);


  // Define a packed struct for representing a floating-point number in super format.
  typedef struct packed {
    logic                      sign;
    logic [SUPER_EXP_BITS-1:0] exponent;
    logic [SUPER_MAN_BITS-1:0] mantissa;
  } fp_t;


  // Internal signals to store the input operands and formats.
  logic                  [           2:0][      WIDTH-1:0]            operands_q;
  fpnew_pkg::fp_format_e                                              src_fmt_q;
  fpnew_pkg::fp_format_e                                              dst_fmt_q;


  // Pipeline registers for the input stage.
  logic                  [0:NUM_INP_REGS][            2:0][WIDTH-1:0] inp_pipe_operands_q;
  logic                  [0:NUM_INP_REGS][NUM_FORMATS-1:0][      2:0] inp_pipe_is_boxed_q;
  fpnew_pkg::roundmode_e [0:NUM_INP_REGS]                             inp_pipe_rnd_mode_q;
  fpnew_pkg::operation_e [0:NUM_INP_REGS]                             inp_pipe_op_q;
  logic                  [0:NUM_INP_REGS]                             inp_pipe_op_mod_q;
  fpnew_pkg::fp_format_e [0:NUM_INP_REGS]                             inp_pipe_src_fmt_q;
  fpnew_pkg::fp_format_e [0:NUM_INP_REGS]                             inp_pipe_dst_fmt_q;
  TagType                [0:NUM_INP_REGS]                             inp_pipe_tag_q;
  logic                  [0:NUM_INP_REGS]                             inp_pipe_mask_q;
  AuxType                [0:NUM_INP_REGS]                             inp_pipe_aux_q;
  logic                  [0:NUM_INP_REGS]                             inp_pipe_valid_q;

  // Ready signal for each stage of the input pipeline.
  logic                  [0:NUM_INP_REGS]                             inp_pipe_ready;


  // Connect the input signals to the first stage of the input pipeline.
  assign inp_pipe_operands_q[0] = operands_i;
  assign inp_pipe_is_boxed_q[0] = is_boxed_i;
  assign inp_pipe_rnd_mode_q[0] = rnd_mode_i;
  assign inp_pipe_op_q[0]       = op_i;
  assign inp_pipe_op_mod_q[0]   = op_mod_i;
  assign inp_pipe_src_fmt_q[0]  = src_fmt_i;
  assign inp_pipe_dst_fmt_q[0]  = dst_fmt_i;
  assign inp_pipe_tag_q[0]      = tag_i;
  assign inp_pipe_mask_q[0]     = mask_i;
  assign inp_pipe_aux_q[0]      = aux_i;
  assign inp_pipe_valid_q[0]    = in_valid_i;

  // Assign the ready signal for the external input.
  assign in_ready_o             = inp_pipe_ready[0];

  // Generate the input pipeline registers.
  for (genvar i = 0; i < NUM_INP_REGS; i++) begin : gen_input_pipeline

    // Register enable signal for the current stage.
    logic reg_ena;

    // Calculate the ready signal for the previous stage.
    assign inp_pipe_ready[i] = inp_pipe_ready[i+1] | ~inp_pipe_valid_q[i+1];

    // Instantiate a register with load enable, clear, and set functionality for the valid bit.
    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        inp_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        inp_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (inp_pipe_ready[i]) ? (inp_pipe_valid_q[i]) : (inp_pipe_valid_q[i+1]);
      end
    end

    // Determine when to enable the registers in this stage.
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

  // Assign the output of the input pipeline to internal signals.
  assign operands_q = inp_pipe_operands_q[NUM_INP_REGS];
  assign src_fmt_q  = inp_pipe_src_fmt_q[NUM_INP_REGS];
  assign dst_fmt_q  = inp_pipe_dst_fmt_q[NUM_INP_REGS];


  // Signals to store the sign, exponent, and mantissa for each format and operand.
  logic                [NUM_FORMATS-1:0][2:0]                     fmt_sign;
  logic signed         [NUM_FORMATS-1:0][2:0][SUPER_EXP_BITS-1:0] fmt_exponent;
  logic                [NUM_FORMATS-1:0][2:0][SUPER_MAN_BITS-1:0] fmt_mantissa;

  // Array to store classification information for each format and operand.
  fpnew_pkg::fp_info_t [NUM_FORMATS-1:0][2:0]                     info_q;


  // Loop through all supported floating-point formats.
  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : fmt_init_inputs

    // Local parameter for the width of the current floating-point format.
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    // Local parameter for the number of exponent bits in the current format.
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    // Local parameter for the number of mantissa bits in the current format.
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    // Process only if the current format is enabled in the configuration.
    if (FpFmtConfig[fmt]) begin : active_format
      // Temporary signal to hold the trimmed operand for the current format.
      logic [2:0][FP_WIDTH-1:0] trimmed_ops;

      // Instantiate the floating-point classifier module.
      fpnew_classifier #(
          .FpFormat   (fpnew_pkg::fp_format_e'(fmt)),
          .NumOperands(3)
      ) i_fpnew_classifier (
          .operands_i(trimmed_ops),
          .is_boxed_i(inp_pipe_is_boxed_q[NUM_INP_REGS][fmt]),
          .info_o    (info_q[fmt])
      );
      // Loop through the three operands.
      for (genvar op = 0; op < 3; op++) begin : gen_operands
        // Trim the operand to the width of the current format.
        assign trimmed_ops[op] = operands_q[op][FP_WIDTH-1:0];
        // Extract the sign bit.
        assign fmt_sign[fmt][op] = operands_q[op][FP_WIDTH-1];
        // Extract and sign-extend the exponent.
        assign fmt_exponent[fmt][op] = signed'({1'b0, operands_q[op][MAN_BITS+:EXP_BITS]});
        // Extract the mantissa and add the implicit leading one for normal numbers.
        assign fmt_mantissa[fmt][op] = {info_q[fmt][op].is_normal, operands_q[op][MAN_BITS-1:0]} <<
                                         (SUPER_MAN_BITS - MAN_BITS);
      end
    end else begin : inactive_format
      // Assign default "don't care" values for inactive formats.
      assign info_q[fmt]       = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_sign[fmt]     = fpnew_pkg::DONT_CARE;
      assign fmt_exponent[fmt] = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_mantissa[fmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end

  // Signals to hold the floating-point representation of the operands.
  fp_t operand_a, operand_b, operand_c;
  // Signals to hold the classification information of the operands.
  fpnew_pkg::fp_info_t info_a, info_b, info_c;


  // Combinational block to select operands based on the source and destination formats.
  always_comb begin : op_select

    // Assign the super format representation of the operands based on the selected source format.
    operand_a = {fmt_sign[src_fmt_q][0], fmt_exponent[src_fmt_q][0], fmt_mantissa[src_fmt_q][0]};
    operand_b = {fmt_sign[src_fmt_q][1], fmt_exponent[src_fmt_q][1], fmt_mantissa[src_fmt_q][1]};
    // Assign the super format representation of the third operand based on the selected destination format.
    operand_c = {fmt_sign[dst_fmt_q][2], fmt_exponent[dst_fmt_q][2], fmt_mantissa[dst_fmt_q][2]};
    // Assign the classification information of the operands.
    info_a = info_q[src_fmt_q][0];
    info_b = info_q[src_fmt_q][1];
    info_c = info_q[dst_fmt_q][2];

    // Apply the operation modifier (typically for negation of the third operand).
    operand_c.sign = operand_c.sign ^ inp_pipe_op_mod_q[NUM_INP_REGS];

    // Perform operand adjustments based on the selected operation.
    unique case (inp_pipe_op_q[NUM_INP_REGS])
      // For FMADD, no specific adjustments are needed at this stage.
      fpnew_pkg::FMADD:  ;
      // For FNMSUB, negate the sign of the first operand.
      fpnew_pkg::FNMSUB: operand_a.sign = ~operand_a.sign;
      // For ADD operation, treat the first operand as zero with the source format's bias.
      fpnew_pkg::ADD: begin
        operand_a = '{sign: 1'b0, exponent: fpnew_pkg::bias(src_fmt_q), mantissa: '0};
        info_a    = '{is_normal: 1'b1, is_boxed: 1'b1, default: 1'b0};
      end
      // For MUL operation, treat the third operand as either positive or negative zero.
      fpnew_pkg::MUL: begin
        if (inp_pipe_rnd_mode_q[NUM_INP_REGS] == fpnew_pkg::RDN)
          operand_c = '{sign: 1'b0, exponent: '0, mantissa: '0};
        else operand_c = '{sign: 1'b1, exponent: '0, mantissa: '0};
        info_c = '{is_zero: 1'b1, is_boxed: 1'b1, default: 1'b0};
      end
      // Default case for unsupported operations.
      default: begin
        operand_a = '{default: fpnew_pkg::DONT_CARE};
        operand_b = '{default: fpnew_pkg::DONT_CARE};
        operand_c = '{default: fpnew_pkg::DONT_CARE};
        info_a    = '{default: fpnew_pkg::DONT_CARE};
        info_b    = '{default: fpnew_pkg::DONT_CARE};
        info_c    = '{default: fpnew_pkg::DONT_CARE};
      end
    endcase
  end


  // Flags to detect special operand conditions.
  logic any_operand_inf;
  logic any_operand_nan;
  logic signalling_nan;
  // Flag indicating if the operation is an effective subtraction.
  logic effective_subtraction;
  // Tentative sign of the result before normalization.
  logic tentative_sign;


  // Determine if any of the operands are infinity.
  assign any_operand_inf = (|{info_a.is_inf, info_b.is_inf, info_c.is_inf});
  // Determine if any of the operands are NaN.
  assign any_operand_nan = (|{info_a.is_nan, info_b.is_nan, info_c.is_nan});
  // Determine if any of the operands are signaling NaN.
  assign signalling_nan = (|{info_a.is_signalling, info_b.is_signalling, info_c.is_signalling});

  // Determine if the effective operation is a subtraction.
  assign effective_subtraction = operand_a.sign ^ operand_b.sign ^ operand_c.sign;

  // Calculate the tentative sign of the result (sign of a * b).
  assign tentative_sign = operand_a.sign ^ operand_b.sign;


  // Signals to store the result and status for special cases.
  logic               [      WIDTH-1:0]            special_result;
  fpnew_pkg::status_t                              special_status;
  logic                                            result_is_special;

  // Signals to store special results and status for each format.
  logic               [NUM_FORMATS-1:0][WIDTH-1:0] fmt_special_result;
  fpnew_pkg::status_t [NUM_FORMATS-1:0]            fmt_special_status;
  logic               [NUM_FORMATS-1:0]            fmt_result_is_special;


  // Loop through all supported floating-point formats to handle special cases.
  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_special_results

    // Local parameters for the width, exponent bits, and mantissa bits of the current format.
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    // Local parameters for quiet NaN exponent and mantissa, and zero mantissa.
    localparam logic [EXP_BITS-1:0] QNAN_EXPONENT = '1;
    localparam logic [MAN_BITS-1:0] QNAN_MANTISSA = 2 ** (MAN_BITS - 1);
    localparam logic [MAN_BITS-1:0] ZERO_MANTISSA = '0;

    // Process only if the current format is enabled.
    if (FpFmtConfig[fmt]) begin : active_format
      // Combinational block to determine special results.
      always_comb begin : special_results
        // Default special result is a quiet NaN.
        logic [FP_WIDTH-1:0] special_res;

        // Initialize special result and status.
        special_res                = {1'b0, QNAN_EXPONENT, QNAN_MANTISSA};
        fmt_special_status[fmt]    = '0;
        fmt_result_is_special[fmt] = 1'b0;


        // Handle invalid operation (NV) for infinity * zero or zero * infinity.
        if ((info_a.is_inf && info_b.is_zero) || (info_a.is_zero && info_b.is_inf)) begin
          fmt_result_is_special[fmt] = 1'b1;
          fmt_special_status[fmt].NV = 1'b1;

          // Handle NaN operands.
        end else if (any_operand_nan) begin
          fmt_result_is_special[fmt] = 1'b1;
          fmt_special_status[fmt].NV = signalling_nan;

          // Handle infinity operands.
        end else if (any_operand_inf) begin
          fmt_result_is_special[fmt] = 1'b1;

          // Invalid operation for (inf * X) +/- inf.
          if ((info_a.is_inf || info_b.is_inf) && info_c.is_inf && effective_subtraction)
            fmt_special_status[fmt].NV = 1'b1;

          // Result is infinity with the sign of (a * b).
          else if (info_a.is_inf || info_b.is_inf) begin
            special_res = {operand_a.sign ^ operand_b.sign, QNAN_EXPONENT, ZERO_MANTISSA};

            // Result is infinity with the sign of c.
          end else if (info_c.is_inf) begin
            special_res = {operand_c.sign, QNAN_EXPONENT, ZERO_MANTISSA};
          end
        end

        // Assign the special result for the current format.
        fmt_special_result[fmt]               = '1;
        fmt_special_result[fmt][FP_WIDTH-1:0] = special_res;
      end
    end else begin : inactive_format
      // Assign default values for inactive formats.
      assign fmt_special_result[fmt] = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_special_status[fmt] = '0;
      assign fmt_result_is_special[fmt] = 1'b0;
    end
  end

  // Select the special result based on the destination format.
  assign result_is_special = fmt_result_is_special[dst_fmt_q];

  // Select the special status based on the destination format.
  assign special_status = fmt_special_status[dst_fmt_q];

  // Select the special result value based on the destination format.
  assign special_result = fmt_special_result[dst_fmt_q];


  // Signals for the exponents of the operands.
  logic signed [EXP_WIDTH-1:0] exponent_a, exponent_b, exponent_c;
  // Signals for intermediate exponent calculations.
  logic signed [EXP_WIDTH-1:0] exponent_addend, exponent_product, exponent_difference;
  // Tentative exponent of the result before normalization.
  logic signed [EXP_WIDTH-1:0] tentative_exponent;


  // Assign the exponents of the operands, sign-extending to EXP_WIDTH.
  assign exponent_a = signed'({1'b0, operand_a.exponent});
  assign exponent_b = signed'({1'b0, operand_b.exponent});
  assign exponent_c = signed'({1'b0, operand_c.exponent});


  // Calculate the exponent of the addend (c).
  assign exponent_addend = signed'(exponent_c + $signed({1'b0, ~info_c.is_normal}));

  // Calculate the exponent of the product (a * b).
  assign exponent_product = (info_a.is_zero || info_b.is_zero) ? 2 - signed'(fpnew_pkg::bias(
      dst_fmt_q
  )) : signed'(exponent_a + info_a.is_subnormal + exponent_b + info_b.is_subnormal -
               2 * signed'(fpnew_pkg::bias(
      src_fmt_q
  )) + signed'(fpnew_pkg::bias(
      dst_fmt_q
  )));

  // Calculate the difference between the addend's and product's exponents.
  assign exponent_difference = exponent_addend - exponent_product;

  // Determine the tentative exponent based on the exponent difference.
  assign tentative_exponent = (exponent_difference > 0) ? exponent_addend : exponent_product;


  // Signal for the shift amount of the addend.
  logic [SHIFT_AMOUNT_WIDTH-1:0] addend_shamt;

  // Combinational block to calculate the shift amount for the addend.
  always_comb begin : addend_shift_amount

    // If the exponent difference is large, shift by the maximum amount.
    if (exponent_difference <= signed'(-2 * PRECISION_BITS - 1))
      addend_shamt = 3 * PRECISION_BITS + 4;

    // Otherwise, calculate the shift amount based on the exponent difference.
    else if (exponent_difference <= signed'(PRECISION_BITS + 2))
      addend_shamt = unsigned'(signed'(PRECISION_BITS) + 3 - exponent_difference);

    // If the exponent difference is small, no shift is needed.
    else
      addend_shamt = 0;
  end


  // Signals for the mantissas of the operands.
  logic [PRECISION_BITS-1:0] mantissa_a, mantissa_b, mantissa_c;
  // Signal for the product of the mantissas.
  logic [2*PRECISION_BITS-1:0] product;
  // Signal for the product shifted by two bits.
  logic [3*PRECISION_BITS+3:0] product_shifted;


  // Assign the mantissas, including the implicit leading one for normal numbers.
  assign mantissa_a = {info_a.is_normal, operand_a.mantissa};
  assign mantissa_b = {info_b.is_normal, operand_b.mantissa};
  assign mantissa_c = {info_c.is_normal, operand_c.mantissa};


  // Calculate the product of the mantissas.
  assign product = mantissa_a * mantissa_b;


  // Shift the product left by two bits to accommodate potential leading zeros.
  assign product_shifted = product << 2;


  // Signal for the shifted addend.
  logic [3*PRECISION_BITS+3:0] addend_after_shift;
  // Signal for the sticky bits of the addend before shifting.
  logic [  PRECISION_BITS-1:0] addend_sticky_bits;
  // Flag indicating if any sticky bits are set before addition.
  logic                        sticky_before_add;
  // Signal for the addend after potential negation.
  logic [3*PRECISION_BITS+3:0] addend_shifted;
  // Carry-in bit for the addition based on effective subtraction.
  logic                        inject_carry_in;


  // Shift the mantissa of the addend based on the calculated shift amount.
  assign {addend_after_shift, addend_sticky_bits} =
      (mantissa_c << (3 * PRECISION_BITS + 4)) >> addend_shamt;

  // Check if any of the shifted-out bits of the addend were non-zero.
  assign sticky_before_add = (|addend_sticky_bits);


  // Negate the shifted addend if it's an effective subtraction.
  assign addend_shifted = (effective_subtraction) ? ~addend_after_shift : addend_after_shift;
  // Inject a carry-in of 1 if it's an effective subtraction and no sticky bits were set.
  assign inject_carry_in = effective_subtraction & ~sticky_before_add;


  // Signal for the raw sum of the product and addend.
  logic [3*PRECISION_BITS+4:0] sum_raw;
  // Carry-out bit from the addition.
  logic                        sum_carry;
  // Signal for the final sum.
  logic [3*PRECISION_BITS+3:0] sum;
  // Sign of the final result.
  logic                        final_sign;


  // Perform the addition of the shifted product and addend with potential carry-in.
  assign sum_raw = product_shifted + addend_shifted + inject_carry_in;
  // Extract the carry-out bit.
  assign sum_carry = sum_raw[3*PRECISION_BITS+4];


  // Adjust the sum if it's an effective subtraction and a borrow occurred.
  assign sum = (effective_subtraction && ~sum_carry) ? -sum_raw : sum_raw;


  // Determine the final sign of the result.
  assign final_sign = (effective_subtraction && (sum_carry == tentative_sign))
                      ? 1'b1
                      : (effective_subtraction ? 1'b0 : tentative_sign);


  // Pipeline registers for the middle stage.
  logic                  [0:NUM_MID_REGS]                         mid_pipe_eff_sub_q;
  logic signed           [0:NUM_MID_REGS][         EXP_WIDTH-1:0] mid_pipe_exp_prod_q;
  logic signed           [0:NUM_MID_REGS][         EXP_WIDTH-1:0] mid_pipe_exp_diff_q;
  logic signed           [0:NUM_MID_REGS][         EXP_WIDTH-1:0] mid_pipe_tent_exp_q;
  logic                  [0:NUM_MID_REGS][SHIFT_AMOUNT_WIDTH-1:0] mid_pipe_add_shamt_q;
  logic                  [0:NUM_MID_REGS]                         mid_pipe_sticky_q;
  logic                  [0:NUM_MID_REGS][  3*PRECISION_BITS+3:0] mid_pipe_sum_q;
  logic                  [0:NUM_MID_REGS]                         mid_pipe_final_sign_q;
  fpnew_pkg::roundmode_e [0:NUM_MID_REGS]                         mid_pipe_rnd_mode_q;
  fpnew_pkg::fp_format_e [0:NUM_MID_REGS]                         mid_pipe_dst_fmt_q;
  logic                  [0:NUM_MID_REGS]                         mid_pipe_res_is_spec_q;
  fp_t                   [0:NUM_MID_REGS]                         mid_pipe_spec_res_q;
  fpnew_pkg::status_t    [0:NUM_MID_REGS]                         mid_pipe_spec_stat_q;
  TagType                [0:NUM_MID_REGS]                         mid_pipe_tag_q;
  logic                  [0:NUM_MID_REGS]                         mid_pipe_mask_q;
  AuxType                [0:NUM_MID_REGS]                         mid_pipe_aux_q;
  logic                  [0:NUM_MID_REGS]                         mid_pipe_valid_q;

  // Ready signal for each stage of the middle pipeline.
  logic                  [0:NUM_MID_REGS]                         mid_pipe_ready;


  // Connect the outputs of the previous stage to the first stage of the middle pipeline.
  assign mid_pipe_eff_sub_q[0]        = effective_subtraction;
  assign mid_pipe_exp_prod_q[0]       = exponent_product;
  assign mid_pipe_exp_diff_q[0]       = exponent_difference;
  assign mid_pipe_tent_exp_q[0]       = tentative_exponent;
  assign mid_pipe_add_shamt_q[0]      = addend_shamt;
  assign mid_pipe_sticky_q[0]         = sticky_before_add;
  assign mid_pipe_sum_q[0]            = sum;
  assign mid_pipe_final_sign_q[0]     = final_sign;
  assign mid_pipe_rnd_mode_q[0]       = inp_pipe_rnd_mode_q[NUM_INP_REGS];
  assign mid_pipe_dst_fmt_q[0]        = dst_fmt_q;
  assign mid_pipe_res_is_spec_q[0]    = result_is_special;
  assign mid_pipe_spec_res_q[0]       = special_result;
  assign mid_pipe_spec_stat_q[0]      = special_status;
  assign mid_pipe_tag_q[0]            = inp_pipe_tag_q[NUM_INP_REGS];
  assign mid_pipe_mask_q[0]           = inp_pipe_mask_q[NUM_INP_REGS];
  assign mid_pipe_aux_q[0]            = inp_pipe_aux_q[NUM_INP_REGS];
  assign mid_pipe_valid_q[0]          = inp_pipe_valid_q[NUM_INP_REGS];

  // Connect the ready signal of the middle pipeline to the previous stage.
  assign inp_pipe_ready[NUM_INP_REGS] = mid_pipe_ready[0];


  // Generate the middle pipeline registers.
  for (genvar i = 0; i < NUM_MID_REGS; i++) begin : gen_inside_pipeline

    // Register enable signal for the current stage.
    logic reg_ena;

    // Calculate the ready signal for the previous stage.
    assign mid_pipe_ready[i] = mid_pipe_ready[i+1] | ~mid_pipe_valid_q[i+1];

    // Instantiate a register with load enable, clear, and set functionality for the valid bit.
    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        mid_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        mid_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (mid_pipe_ready[i]) ? (mid_pipe_valid_q[i]) : (mid_pipe_valid_q[i+1]);
      end
    end

    // Determine when to enable the registers in this stage.
    assign reg_ena = (mid_pipe_ready[i] & mid_pipe_valid_q[i]) | reg_ena_i[NUM_INP_REGS+i];

    // Instantiate registers for the intermediate signals.
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_eff_sub_q[i+1] <= ('0);
      end else begin
        mid_pipe_eff_sub_q[i+1] <= (reg_ena) ? (mid_pipe_eff_sub_q[i]) : (mid_pipe_eff_sub_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_exp_prod_q[i+1] <= ('0);
      end else begin
        mid_pipe_exp_prod_q[i+1] <= (reg_ena) ? (mid_pipe_exp_prod_q[i]) : (mid_pipe_exp_prod_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_exp_diff_q[i+1] <= ('0);
      end else begin
        mid_pipe_exp_diff_q[i+1] <= (reg_ena) ? (mid_pipe_exp_diff_q[i]) : (mid_pipe_exp_diff_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_tent_exp_q[i+1] <= ('0);
      end else begin
        mid_pipe_tent_exp_q[i+1] <= (reg_ena) ? (mid_pipe_tent_exp_q[i]) : (mid_pipe_tent_exp_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_add_shamt_q[i+1] <= ('0);
      end else begin
        mid_pipe_add_shamt_q[i+1] <= (reg_ena) ? (mid_pipe_add_shamt_q[i]) : (mid_pipe_add_shamt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_sticky_q[i+1] <= ('0);
      end else begin
        mid_pipe_sticky_q[i+1] <= (reg_ena) ? (mid_pipe_sticky_q[i]) : (mid_pipe_sticky_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_sum_q[i+1] <= ('0);
      end else begin
        mid_pipe_sum_q[i+1] <= (reg_ena) ? (mid_pipe_sum_q[i]) : (mid_pipe_sum_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_final_sign_q[i+1] <= ('0);
      end else begin
        mid_pipe_final_sign_q[i+1] <= (reg_ena) ? (mid_pipe_final_sign_q[i]) : (mid_pipe_final_sign_q[i+1]);
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
        mid_pipe_dst_fmt_q[i+1] <= (fpnew_pkg::fp_format_e'(0));
      end else begin
        mid_pipe_dst_fmt_q[i+1] <= (reg_ena) ? (mid_pipe_dst_fmt_q[i]) : (mid_pipe_dst_fmt_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_res_is_spec_q[i+1] <= ('0);
      end else begin
        mid_pipe_res_is_spec_q[i+1] <= (reg_ena) ? (mid_pipe_res_is_spec_q[i]) : (mid_pipe_res_is_spec_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_spec_res_q[i+1] <= ('0);
      end else begin
        mid_pipe_spec_res_q[i+1] <= (reg_ena) ? (mid_pipe_spec_res_q[i]) : (mid_pipe_spec_res_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mid_pipe_spec_stat_q[i+1] <= ('0);
      end else begin
        mid_pipe_spec_stat_q[i+1] <= (reg_ena) ? (mid_pipe_spec_stat_q[i]) : (mid_pipe_spec_stat_q[i+1]);
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

  // Assign the output of the middle pipeline to internal signals.
  logic                                           effective_subtraction_q;
  logic signed           [         EXP_WIDTH-1:0] exponent_product_q;
  logic signed           [         EXP_WIDTH-1:0] exponent_difference_q;
  logic signed           [         EXP_WIDTH-1:0] tentative_exponent_q;
  logic                  [SHIFT_AMOUNT_WIDTH-1:0] addend_shamt_q;
  logic                                           sticky_before_add_q;
  logic                  [  3*PRECISION_BITS+3:0] sum_q;
  logic                                           final_sign_q;
  fpnew_pkg::fp_format_e                          dst_fmt_q2;
  fpnew_pkg::roundmode_e                          rnd_mode_q;
  logic                                           result_is_special_q;
  fp_t                                            special_result_q;
  fpnew_pkg::status_t                             special_status_q;

  assign effective_subtraction_q = mid_pipe_eff_sub_q[NUM_MID_REGS];
  assign exponent_product_q      = mid_pipe_exp_prod_q[NUM_MID_REGS];
  assign exponent_difference_q   = mid_pipe_exp_diff_q[NUM_MID_REGS];
  assign tentative_exponent_q    = mid_pipe_tent_exp_q[NUM_MID_REGS];
  assign addend_shamt_q          = mid_pipe_add_shamt_q[NUM_MID_REGS];
  assign sticky_before_add_q     = mid_pipe_sticky_q[NUM_MID_REGS];
  assign sum_q                   = mid_pipe_sum_q[NUM_MID_REGS];
  assign final_sign_q            = mid_pipe_final_sign_q[NUM_MID_REGS];
  assign rnd_mode_q              = mid_pipe_rnd_mode_q[NUM_MID_REGS];
  assign dst_fmt_q2              = mid_pipe_dst_fmt_q[NUM_MID_REGS];
  assign result_is_special_q     = mid_pipe_res_is_spec_q[NUM_MID_REGS];
  assign special_result_q        = mid_pipe_spec_res_q[NUM_MID_REGS];
  assign special_status_q        = mid_pipe_spec_stat_q[NUM_MID_REGS];


  // Signal for the lower part of the sum for leading zero counting.
  logic        [   LOWER_SUM_WIDTH-1:0] sum_lower;
  // Signal for the count of leading zeros.
  logic        [  LZC_RESULT_WIDTH-1:0] leading_zero_count;
  // Signed version of the leading zero count.
  logic signed [    LZC_RESULT_WIDTH:0] leading_zero_count_sgn;
  // Flag indicating if all bits in the lower sum are zero.
  logic                                 lzc_zeroes;

  // Signal for the normalization shift amount.
  logic        [SHIFT_AMOUNT_WIDTH-1:0] norm_shamt;
  // Signal for the normalized exponent.
  logic signed [         EXP_WIDTH-1:0] normalized_exponent;

  // Signal for the sum after normalization shift.
  logic        [  3*PRECISION_BITS+4:0] sum_shifted;
  // Signal for the final mantissa after normalization.
  logic        [      PRECISION_BITS:0] final_mantissa;
  // Signal for the sticky bits after normalization.
  logic        [  2*PRECISION_BITS+2:0] sum_sticky_bits;
  // Flag indicating if any sticky bits are set after normalization.
  logic                                 sticky_after_norm;

  // Signal for the final exponent after normalization.
  logic signed [         EXP_WIDTH-1:0] final_exponent;

  // Extract the lower part of the sum for leading zero counting.
  assign sum_lower = sum_q[LOWER_SUM_WIDTH-1:0];


  // Instantiate the leading zero counter module.
  lzc #(
      .WIDTH(LOWER_SUM_WIDTH),
      .MODE (1)
  ) i_lzc (
      .in_i   (sum_lower),
      .cnt_o  (leading_zero_count),
      .empty_o(lzc_zeroes)
  );

  // Sign-extend the leading zero count.
  assign leading_zero_count_sgn = signed'({1'b0, leading_zero_count});


  // Combinational block to determine the normalization shift amount and exponent.
  always_comb begin : norm_shift_amount

    // Handle cases where the exponent difference is small or it's an effective subtraction.
    if ((exponent_difference_q <= 0) || (effective_subtraction_q && (exponent_difference_q <= 2))) begin

      // If the exponent after normalization is non-negative and not all lower sum bits are zero.
      if ((exponent_product_q - leading_zero_count_sgn + 1 >= 0) && !lzc_zeroes) begin

        norm_shamt          = PRECISION_BITS + 2 + leading_zero_count;
        normalized_exponent = exponent_product_q - leading_zero_count_sgn + 1;

        // Otherwise, handle subnormal numbers.
      end else begin

        norm_shamt          = unsigned'(signed'(PRECISION_BITS + 2 + exponent_product_q));
        normalized_exponent = 0;
      end

      // Handle cases where the exponent difference is larger.
    end else begin
      norm_shamt          = addend_shamt_q;
      normalized_exponent = tentative_exponent_q;
    end
  end


  // Shift the sum based on the normalization shift amount.
  assign sum_shifted = sum_q << norm_shamt;


  // Combinational block for small normalization adjustments.
  always_comb begin : small_norm

    // Extract the final mantissa and sticky bits.
    {final_mantissa, sum_sticky_bits} = sum_shifted;
    // Assign the normalized exponent.
    final_exponent                    = normalized_exponent;


    // Handle potential overflow during normalization.
    if (sum_shifted[3*PRECISION_BITS+4]) begin
      {final_mantissa, sum_sticky_bits} = sum_shifted >> 1;
      final_exponent                    = normalized_exponent + 1;

      // Handle cases where the most significant bit is set.
    end else if (sum_shifted[3*PRECISION_BITS+3]) begin


      // Handle cases where the normalized exponent is greater than 1.
    end else if (normalized_exponent > 1) begin
      {final_mantissa, sum_sticky_bits} = sum_shifted << 1;
      final_exponent                    = normalized_exponent - 1;

      // Handle cases where the normalized exponent is not greater than 1.
    end else begin
      final_exponent = '0;
    end
  end


  // Calculate the final sticky bit after normalization.
  assign sticky_after_norm = (|{sum_sticky_bits}) | sticky_before_add_q;


  // Signals for pre-rounding values.
  logic                                     pre_round_sign;
  logic [SUPER_EXP_BITS+SUPER_MAN_BITS-1:0] pre_round_abs;
  // Sticky bits for rounding.
  logic [                              1:0] round_sticky_bits;

  // Flags for overflow and underflow before and after rounding.
  logic of_before_round, of_after_round;
  logic uf_before_round, uf_after_round;

  // Signals for pre-rounding absolute value and sticky bits for each format.
  logic [NUM_FORMATS-1:0][SUPER_EXP_BITS+SUPER_MAN_BITS-1:0] fmt_pre_round_abs;
  logic [NUM_FORMATS-1:0][1:0] fmt_round_sticky_bits;

  // Flags for overflow and underflow after rounding for each format.
  logic [NUM_FORMATS-1:0] fmt_of_after_round;
  logic [NUM_FORMATS-1:0] fmt_uf_after_round;

  // Signals for rounded result.
  logic rounded_sign;
  logic [SUPER_EXP_BITS+SUPER_MAN_BITS-1:0] rounded_abs;
  // Flag indicating if the result is exactly zero.
  logic result_zero;


  // Determine overflow and underflow before rounding.
  assign of_before_round = final_exponent >= 2 ** (fpnew_pkg::exp_bits(dst_fmt_q2)) - 1;
  assign uf_before_round = final_exponent == 0;


  // Loop through all supported floating-point formats for result assembly.
  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_res_assemble

    // Local parameters for exponent and mantissa bits of the current format.
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    // Signals for pre-rounding exponent and mantissa for the current format.
    logic [EXP_BITS-1:0] pre_round_exponent;
    logic [MAN_BITS-1:0] pre_round_mantissa;

    // Process only if the current format is enabled.
    if (FpFmtConfig[fmt]) begin : active_format

      // Determine pre-rounding exponent and mantissa, handling potential overflow.
      assign pre_round_exponent = (of_before_round) ? 2**EXP_BITS-2 : final_exponent[EXP_BITS-1:0];
      assign pre_round_mantissa = (of_before_round) ? '1 : final_mantissa[SUPER_MAN_BITS-:MAN_BITS];

      // Combine pre-rounding exponent and mantissa into the absolute value.
      assign fmt_pre_round_abs[fmt] = {pre_round_exponent, pre_round_mantissa};


      // Determine the second sticky bit for rounding.
      assign fmt_round_sticky_bits[fmt][1] = final_mantissa[SUPER_MAN_BITS-MAN_BITS] |
                                               of_before_round;


      // Determine the first sticky bit for rounding, handling narrow formats.
      if (MAN_BITS < SUPER_MAN_BITS) begin : narrow_sticky
        assign fmt_round_sticky_bits[fmt][0] = (| final_mantissa[SUPER_MAN_BITS-MAN_BITS-1:0]) |
                                                 sticky_after_norm | of_before_round;
      end else begin : normal_sticky
        assign fmt_round_sticky_bits[fmt][0] = sticky_after_norm | of_before_round;
      end
    end else begin : inactive_format
      // Assign default values for inactive formats.
      assign fmt_pre_round_abs[fmt] = '{default: fpnew_pkg::DONT_CARE};
      assign fmt_round_sticky_bits[fmt] = '{default: fpnew_pkg::DONT_CARE};
    end
  end


  // Assign pre-rounding sign and absolute value.
  assign pre_round_sign   = final_sign_q;
  assign pre_round_abs    = fmt_pre_round_abs[dst_fmt_q2];


  // Assign rounding sticky bits.
  assign round_sticky_bits = fmt_round_sticky_bits[dst_fmt_q2];


  // Instantiate the floating-point rounding module.
  fpnew_rounding #(
      .AbsWidth(SUPER_EXP_BITS + SUPER_MAN_BITS)
  ) i_fpnew_rounding (
      .abs_value_i            (pre_round_abs),
      .sign_i                 (pre_round_sign),
      .round_sticky_bits_i    (round_sticky_bits),
      .rnd_mode_i             (rnd_mode_q),
      .effective_subtraction_i(effective_subtraction_q),
      .abs_rounded_o          (rounded_abs),
      .sign_o                 (rounded_sign),
      .exact_zero_o           (result_zero)
  );

  // Signal to store the final result for each format.
  logic [NUM_FORMATS-1:0][WIDTH-1:0] fmt_result;

  // Loop through all supported floating-point formats for sign injection.
  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_sign_inject

    // Local parameters for width, exponent bits, and mantissa bits of the current format.
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(fpnew_pkg::fp_format_e'(fmt));
    localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(fpnew_pkg::fp_format_e'(fmt));

    // Process only if the current format is enabled.
    if (FpFmtConfig[fmt]) begin : active_format
      // Combinational block for post-processing after rounding.
      always_comb begin : post_process

        // Determine underflow after rounding.
        fmt_uf_after_round[fmt] = (rounded_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == '0)
        || ((pre_round_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == '0) && (rounded_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == 1) &&
            ((round_sticky_bits != 2'b11) || (!sum_sticky_bits[MAN_BITS*2 + 4] && ((rnd_mode_i == fpnew_pkg::RNE) || (rnd_mode_i == fpnew_pkg::RMM)))));
        // Determine overflow after rounding.
        fmt_of_after_round[fmt] = rounded_abs[EXP_BITS+MAN_BITS-1:MAN_BITS] == '1;


        // Assemble the final result with the rounded sign and absolute value.
        fmt_result[fmt] = '1;
        fmt_result[fmt][FP_WIDTH-1:0] = {rounded_sign, rounded_abs[EXP_BITS+MAN_BITS-1:0]};
      end
    end else begin : inactive_format
      // Assign default values for inactive formats.
      assign fmt_uf_after_round[fmt] = fpnew_pkg::DONT_CARE;
      assign fmt_of_after_round[fmt] = fpnew_pkg::DONT_CARE;
      assign fmt_result[fmt]         = '{default: fpnew_pkg::DONT_CARE};
    end
  end


  // Assign underflow and overflow flags after rounding.
  assign uf_after_round = fmt_uf_after_round[dst_fmt_q2];
  assign of_after_round = fmt_of_after_round[dst_fmt_q2];


  // Signals for the regular result and status.
  logic               [WIDTH-1:0] regular_result;
  fpnew_pkg::status_t             regular_status;


  // Select the result based on the destination format.
  assign regular_result = fmt_result[dst_fmt_q2];
  // Initialize the status flags.
  assign regular_status.NV = 1'b0;
  assign regular_status.DZ = 1'b0;
  // Set overflow flag.
  assign regular_status.OF = of_before_round | of_after_round;
  // Set underflow flag.
  assign regular_status.UF = uf_after_round & regular_status.NX;
  // Set inexact flag.
  assign regular_status.NX = (|round_sticky_bits) | of_before_round | of_after_round;


  // Signals for the final result and status.
  logic               [WIDTH-1:0] result_d;
  fpnew_pkg::status_t             status_d;


  // Select either the special result or the regular result based on the flag.
  assign result_d = result_is_special_q ? special_result_q : regular_result;
  // Select either the special status or the regular status based on the flag.
  assign status_d = result_is_special_q ? special_status_q : regular_status;


  // Pipeline registers for the output stage.
  logic               [0:NUM_OUT_REGS][WIDTH-1:0] out_pipe_result_q;
  fpnew_pkg::status_t [0:NUM_OUT_REGS]            out_pipe_status_q;
  TagType             [0:NUM_OUT_REGS]            out_pipe_tag_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_mask_q;
  AuxType             [0:NUM_OUT_REGS]            out_pipe_aux_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_valid_q;

  // Ready signal for each stage of the output pipeline.
  logic               [0:NUM_OUT_REGS]            out_pipe_ready;


  // Connect the outputs of the previous stage to the first stage of the output pipeline.
  assign out_pipe_result_q[0] = result_d;
  assign out_pipe_status_q[0] = status_d;
  assign out_pipe_tag_q[0]    = mid_pipe_tag_q[NUM_MID_REGS];
  assign out_pipe_mask_q[0]   = mid_pipe_mask_q[NUM_MID_REGS];
  assign out_pipe_aux_q[0]    = mid_pipe_aux_q[NUM_MID_REGS];
  assign out_pipe_valid_q[0]  = mid_pipe_valid_q[NUM_MID_REGS];

  // Connect the ready signal of the output pipeline to the previous stage.
  assign mid_pipe_ready[NUM_MID_REGS] = out_pipe_ready[0];

  // Generate the output pipeline registers.
  for (genvar i = 0; i < NUM_OUT_REGS; i++) begin : gen_output_pipeline

    // Register enable signal for the current stage.
    logic reg_ena;

    // Calculate the ready signal for the previous stage.
    assign out_pipe_ready[i] = out_pipe_ready[i+1] | ~out_pipe_valid_q[i+1];

    // Instantiate a register with load enable, clear, and set functionality for the valid bit.
    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        out_pipe_valid_q[i+1] <= (1'b0);
      end else begin
        out_pipe_valid_q[i+1] <= (flush_i) ? (1'b0) : (out_pipe_ready[i]) ? (out_pipe_valid_q[i]) : (out_pipe_valid_q[i+1]);
      end
    end

    // Determine when to enable the registers in this stage.
    assign reg_ena = (out_pipe_ready[i] & out_pipe_valid_q[i]) | reg_ena_i[NUM_INP_REGS + NUM_MID_REGS + i];

    // Instantiate registers for the output signals.
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

  // Connect the ready signal of the external output to the last stage of the pipeline.
  assign out_pipe_ready[NUM_OUT_REGS] = out_ready_i;

  // Assign the output signals from the last stage of the pipeline.
  assign result_o                     = out_pipe_result_q[NUM_OUT_REGS];
  assign status_o                     = out_pipe_status_q[NUM_OUT_REGS];
  assign extension_bit_o              = 1'b1;
  assign tag_o                        = out_pipe_tag_q[NUM_OUT_REGS];
  assign mask_o                       = out_pipe_mask_q[NUM_OUT_REGS];
  assign aux_o                        = out_pipe_aux_q[NUM_OUT_REGS];
  assign out_valid_o                  = out_pipe_valid_q[NUM_OUT_REGS];
  // Determine if the module is busy.
  assign busy_o                       = (|{inp_pipe_valid_q, mid_pipe_valid_q, out_pipe_valid_q});
endmodule
