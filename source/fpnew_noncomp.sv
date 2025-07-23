module fpnew_noncomp #(
    // Floating-point format for the operation
    parameter fpnew_pkg::fp_format_e   FpFormat    = fpnew_pkg::fp_format_e'(0),
    // Number of pipeline registers
    parameter int unsigned             NumPipeRegs = 0,
    // Pipeline configuration (BEFORE, AFTER, DISTRIBUTED, etc.)
    parameter fpnew_pkg::pipe_config_t PipeConfig  = fpnew_pkg::BEFORE,
    // Type for tagging operations
    parameter type                     TagType     = logic,
    // Type for auxiliary data
    parameter type                     AuxType     = logic,

    // Local parameter for data width
    localparam int unsigned WIDTH = fpnew_pkg::fp_width(FpFormat),
    // Width of the enable signal for pipeline registers
    localparam int unsigned ExtRegEnaWidth = NumPipeRegs == 0 ? 1 : NumPipeRegs
) (
    // Clock signal
    input logic clk_i,
    // Active-low reset signal
    input logic rst_ni,

    // Input operands (two floating-point values)
    input logic                  [1:0][WIDTH-1:0] operands_i,
    // Indicates if the operands are boxed
    input logic                  [1:0]            is_boxed_i,
    // Rounding mode for the operation
    input fpnew_pkg::roundmode_e                  rnd_mode_i,
    // Operation type (e.g., SGNJ, MINMAX, CMP, CLASSIFY)
    input fpnew_pkg::operation_e                  op_i,
    // Modifier for the operation
    input logic                                   op_mod_i,
    // Tag associated with the operation
    input TagType                                 tag_i,
    // Mask signal for the operation
    input logic                                   mask_i,
    // Auxiliary data for the operation
    input AuxType                                 aux_i,

    // Input valid signal
    input  logic in_valid_i,
    // Output ready signal for input stage
    output logic in_ready_o,
    // Flush signal to clear the pipeline
    input  logic flush_i,

    // Result of the operation
    output logic                  [WIDTH-1:0] result_o,
    // Status of the operation
    output fpnew_pkg::status_t                status_o,
    // Extension bit for additional information
    output logic                              extension_bit_o,
    // Class mask for CLASSIFY operation
    output fpnew_pkg::classmask_e             class_mask_o,
    // Indicates if the operation is CLASSIFY
    output logic                              is_class_o,
    // Tag associated with the result
    output TagType                            tag_o,
    // Mask signal for the result
    output logic                              mask_o,
    // Auxiliary data for the result
    output AuxType                            aux_o,

    // Output valid signal
    output logic out_valid_o,
    // Input ready signal for output stage
    input  logic out_ready_i,

    // Busy signal indicating ongoing operation
    output logic busy_o,

    // Enable signals for pipeline registers
    input logic [ExtRegEnaWidth-1:0] reg_ena_i
);

  // Local parameters for exponent and mantissa bits
  localparam int unsigned EXP_BITS = fpnew_pkg::exp_bits(FpFormat);
  localparam int unsigned MAN_BITS = fpnew_pkg::man_bits(FpFormat);

  // Number of input and output pipeline registers based on configuration
  localparam NUM_INP_REGS = (PipeConfig == fpnew_pkg::BEFORE || PipeConfig == fpnew_pkg::INSIDE)
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? ((NumPipeRegs + 1) / 2)
                               : 0);
  localparam NUM_OUT_REGS = PipeConfig == fpnew_pkg::AFTER
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? (NumPipeRegs / 2)
                               : 0);

  // Floating-point structure for operands
  typedef struct packed {
    logic                sign;
    logic [EXP_BITS-1:0] exponent;
    logic [MAN_BITS-1:0] mantissa;
  } fp_t;

  // Input pipeline registers for operands, rounding mode, operation, etc.
  logic                  [0:NUM_INP_REGS][1:0][WIDTH-1:0] inp_pipe_operands_q;
  logic                  [0:NUM_INP_REGS][1:0]            inp_pipe_is_boxed_q;
  fpnew_pkg::roundmode_e [0:NUM_INP_REGS]                 inp_pipe_rnd_mode_q;
  fpnew_pkg::operation_e [0:NUM_INP_REGS]                 inp_pipe_op_q;
  logic                  [0:NUM_INP_REGS]                 inp_pipe_op_mod_q;
  TagType                [0:NUM_INP_REGS]                 inp_pipe_tag_q;
  logic                  [0:NUM_INP_REGS]                 inp_pipe_mask_q;
  AuxType                [0:NUM_INP_REGS]                 inp_pipe_aux_q;
  logic                  [0:NUM_INP_REGS]                 inp_pipe_valid_q;

  logic                  [0:NUM_INP_REGS]                 inp_pipe_ready;

  assign inp_pipe_operands_q[0] = operands_i;
  assign inp_pipe_is_boxed_q[0] = is_boxed_i;
  assign inp_pipe_rnd_mode_q[0] = rnd_mode_i;
  assign inp_pipe_op_q[0]       = op_i;
  assign inp_pipe_op_mod_q[0]   = op_mod_i;
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
        inp_pipe_valid_q[i+1] <= ('0);
      end else begin
        inp_pipe_valid_q[i+1] <= (flush_i) ? ('0) : (inp_pipe_ready[i]) ? (inp_pipe_valid_q[i]) : (inp_pipe_valid_q[i+1]);
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

  fpnew_pkg::fp_info_t [1:0] info_q;

  fpnew_classifier #(
      .FpFormat   (FpFormat),
      .NumOperands(2)
  ) i_class_a (
      .operands_i(inp_pipe_operands_q[NUM_INP_REGS]),
      .is_boxed_i(inp_pipe_is_boxed_q[NUM_INP_REGS]),
      .info_o    (info_q)
  );

  fp_t operand_a, operand_b;
  fpnew_pkg::fp_info_t info_a, info_b;

  assign operand_a = inp_pipe_operands_q[NUM_INP_REGS][0];
  assign operand_b = inp_pipe_operands_q[NUM_INP_REGS][1];
  assign info_a    = info_q[0];
  assign info_b    = info_q[1];

  logic any_operand_inf;
  logic any_operand_nan;
  logic signalling_nan;

  assign any_operand_inf = (|{info_a.is_inf, info_b.is_inf});
  assign any_operand_nan = (|{info_a.is_nan, info_b.is_nan});
  assign signalling_nan  = (|{info_a.is_signalling, info_b.is_signalling});

  logic operands_equal, operand_a_smaller;

  assign operands_equal    = (operand_a == operand_b) || (info_a.is_zero && info_b.is_zero);

  assign operand_a_smaller = (operand_a < operand_b) ^ (operand_a.sign || operand_b.sign);

  fp_t                sgnj_result;
  fpnew_pkg::status_t sgnj_status;
  logic               sgnj_extension_bit;

  always_comb begin : sign_injections
    logic sign_a, sign_b;

    sgnj_result = operand_a;

    if (!info_a.is_boxed)
      sgnj_result = '{sign: 1'b0, exponent: '1, mantissa: 2 ** (MAN_BITS - 1)};

    sign_a = operand_a.sign & info_a.is_boxed;
    sign_b = operand_b.sign & info_b.is_boxed;

    unique case (inp_pipe_rnd_mode_q[NUM_INP_REGS])
      fpnew_pkg::RNE: sgnj_result.sign = sign_b;
      fpnew_pkg::RTZ: sgnj_result.sign = ~sign_b;
      fpnew_pkg::RDN: sgnj_result.sign = sign_a ^ sign_b;
      fpnew_pkg::RUP: sgnj_result = operand_a;
      default:        sgnj_result = '{default: fpnew_pkg::DONT_CARE};
    endcase
  end

  assign sgnj_status = '0;

  assign sgnj_extension_bit = inp_pipe_op_mod_q[NUM_INP_REGS] ? sgnj_result.sign : 1'b1;

  fp_t                minmax_result;
  fpnew_pkg::status_t minmax_status;
  logic               minmax_extension_bit;

  always_comb begin : min_max

    minmax_status = '0;

    minmax_status.NV = signalling_nan;

    if (info_a.is_nan && info_b.is_nan)
      minmax_result = '{sign: 1'b0, exponent: '1, mantissa: 2 ** (MAN_BITS - 1)};

    else if (info_a.is_nan) minmax_result = operand_b;
    else if (info_b.is_nan) minmax_result = operand_a;

    else begin
      unique case (inp_pipe_rnd_mode_q[NUM_INP_REGS])
        fpnew_pkg::RNE: minmax_result = operand_a_smaller ? operand_a : operand_b;
        fpnew_pkg::RTZ: minmax_result = operand_a_smaller ? operand_b : operand_a;
        default: minmax_result = '{default: fpnew_pkg::DONT_CARE};
      endcase
    end
  end

  assign minmax_extension_bit = 1'b1;

  fp_t                cmp_result;
  fpnew_pkg::status_t cmp_status;
  logic               cmp_extension_bit;

  always_comb begin : comparisons

    cmp_result = '0;
    cmp_status = '0;

    if (signalling_nan) cmp_status.NV = 1'b1;

    else begin
      unique case (inp_pipe_rnd_mode_q[NUM_INP_REGS])
        fpnew_pkg::RNE: begin
          if (any_operand_nan) cmp_status.NV = 1'b1;
          else cmp_result = (operand_a_smaller | operands_equal) ^ inp_pipe_op_mod_q[NUM_INP_REGS];
        end
        fpnew_pkg::RTZ: begin
          if (any_operand_nan) cmp_status.NV = 1'b1;
          else cmp_result = (operand_a_smaller & ~operands_equal) ^ inp_pipe_op_mod_q[NUM_INP_REGS];
        end
        fpnew_pkg::RDN: begin
          if (any_operand_nan) cmp_result = inp_pipe_op_mod_q[NUM_INP_REGS];
          else cmp_result = operands_equal ^ inp_pipe_op_mod_q[NUM_INP_REGS];
        end
        default: cmp_result = '{default: fpnew_pkg::DONT_CARE};
      endcase
    end
  end

  assign cmp_extension_bit = 1'b0;

  fpnew_pkg::status_t    class_status;
  logic                  class_extension_bit;
  fpnew_pkg::classmask_e class_mask_d;

  always_comb begin : classify
    if (info_a.is_normal) begin
      class_mask_d = operand_a.sign ? fpnew_pkg::NEGNORM : fpnew_pkg::POSNORM;
    end else if (info_a.is_subnormal) begin
      class_mask_d = operand_a.sign ? fpnew_pkg::NEGSUBNORM : fpnew_pkg::POSSUBNORM;
    end else if (info_a.is_zero) begin
      class_mask_d = operand_a.sign ? fpnew_pkg::NEGZERO : fpnew_pkg::POSZERO;
    end else if (info_a.is_inf) begin
      class_mask_d = operand_a.sign ? fpnew_pkg::NEGINF : fpnew_pkg::POSINF;
    end else if (info_a.is_nan) begin
      class_mask_d = info_a.is_signalling ? fpnew_pkg::SNAN : fpnew_pkg::QNAN;
    end else begin
      class_mask_d = fpnew_pkg::QNAN;
    end
  end

  assign class_status        = '0;
  assign class_extension_bit = 1'b0;

  fp_t                result_d;
  fpnew_pkg::status_t status_d;
  logic               extension_bit_d;
  logic               is_class_d;

  always_comb begin : select_result
    unique case (inp_pipe_op_q[NUM_INP_REGS])
      fpnew_pkg::SGNJ: begin
        result_d        = sgnj_result;
        status_d        = sgnj_status;
        extension_bit_d = sgnj_extension_bit;
      end
      fpnew_pkg::MINMAX: begin
        result_d        = minmax_result;
        status_d        = minmax_status;
        extension_bit_d = minmax_extension_bit;
      end
      fpnew_pkg::CMP: begin
        result_d        = cmp_result;
        status_d        = cmp_status;
        extension_bit_d = cmp_extension_bit;
      end
      fpnew_pkg::CLASSIFY: begin
        result_d        = '{default: fpnew_pkg::DONT_CARE};
        status_d        = class_status;
        extension_bit_d = class_extension_bit;
      end
      default: begin
        result_d        = '{default: fpnew_pkg::DONT_CARE};
        status_d        = '{default: fpnew_pkg::DONT_CARE};
        extension_bit_d = fpnew_pkg::DONT_CARE;
      end
    endcase
  end

  assign is_class_d = (inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::CLASSIFY);

  // Output pipeline registers for result, status, etc.
  fp_t                   [0:NUM_OUT_REGS] out_pipe_result_q;
  fpnew_pkg::status_t    [0:NUM_OUT_REGS] out_pipe_status_q;
  logic                  [0:NUM_OUT_REGS] out_pipe_extension_bit_q;
  fpnew_pkg::classmask_e [0:NUM_OUT_REGS] out_pipe_class_mask_q;
  logic                  [0:NUM_OUT_REGS] out_pipe_is_class_q;
  TagType                [0:NUM_OUT_REGS] out_pipe_tag_q;
  logic                  [0:NUM_OUT_REGS] out_pipe_mask_q;
  AuxType                [0:NUM_OUT_REGS] out_pipe_aux_q;
  logic                  [0:NUM_OUT_REGS] out_pipe_valid_q;

  logic                  [0:NUM_OUT_REGS] out_pipe_ready;

  assign out_pipe_result_q[0]         = result_d;
  assign out_pipe_status_q[0]         = status_d;
  assign out_pipe_extension_bit_q[0]  = extension_bit_d;
  assign out_pipe_class_mask_q[0]     = class_mask_d;
  assign out_pipe_is_class_q[0]       = is_class_d;
  assign out_pipe_tag_q[0]            = inp_pipe_tag_q[NUM_INP_REGS];
  assign out_pipe_mask_q[0]           = inp_pipe_mask_q[NUM_INP_REGS];
  assign out_pipe_aux_q[0]            = inp_pipe_aux_q[NUM_INP_REGS];
  assign out_pipe_valid_q[0]          = inp_pipe_valid_q[NUM_INP_REGS];

  assign inp_pipe_ready[NUM_INP_REGS] = out_pipe_ready[0];

  // Generate output pipeline stages
  for (genvar i = 0; i < NUM_OUT_REGS; i++) begin : gen_output_pipeline
    logic reg_ena;

    assign out_pipe_ready[i] = out_pipe_ready[i+1] | ~out_pipe_valid_q[i+1];

    always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
      if (!rst_ni) begin
        out_pipe_valid_q[i+1] <= ('0);
      end else begin
        out_pipe_valid_q[i+1] <= (flush_i) ? ('0) : (out_pipe_ready[i]) ? (out_pipe_valid_q[i]) : (out_pipe_valid_q[i+1]);
      end
    end

    assign reg_ena = (out_pipe_ready[i] & out_pipe_valid_q[i]) | reg_ena_i[NUM_INP_REGS+i];

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
        out_pipe_extension_bit_q[i+1] <= ('0);
      end else begin
        out_pipe_extension_bit_q[i+1] <= (reg_ena) ? (out_pipe_extension_bit_q[i]) : (out_pipe_extension_bit_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_class_mask_q[i+1] <= (fpnew_pkg::QNAN);
      end else begin
        out_pipe_class_mask_q[i+1] <= (reg_ena) ? (out_pipe_class_mask_q[i]) : (out_pipe_class_mask_q[i+1]);
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        out_pipe_is_class_q[i+1] <= ('0);
      end else begin
        out_pipe_is_class_q[i+1] <= (reg_ena) ? (out_pipe_is_class_q[i]) : (out_pipe_is_class_q[i+1]);
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
  assign extension_bit_o              = out_pipe_extension_bit_q[NUM_OUT_REGS];
  assign class_mask_o                 = out_pipe_class_mask_q[NUM_OUT_REGS];
  assign is_class_o                   = out_pipe_is_class_q[NUM_OUT_REGS];
  assign tag_o                        = out_pipe_tag_q[NUM_OUT_REGS];
  assign mask_o                       = out_pipe_mask_q[NUM_OUT_REGS];
  assign aux_o                        = out_pipe_aux_q[NUM_OUT_REGS];
  assign out_valid_o                  = out_pipe_valid_q[NUM_OUT_REGS];
  assign busy_o                       = (|{inp_pipe_valid_q, out_pipe_valid_q});
endmodule
