// Module definition for a multi-format slice operation group.
module fpnew_opgroup_multifmt_slice #(
    // Parameter for the operation group type (default: CONV).
    parameter fpnew_pkg::opgroup_e OpGroup = fpnew_pkg::CONV,
    // Parameter for the data path width (default: 64 bits).
    parameter int unsigned         Width   = 64,

    // Parameter for the floating-point format configuration (default: all enabled).
    parameter fpnew_pkg::fmt_logic_t   FpFmtConfig   = '1,
    // Parameter for the integer format configuration (default: all enabled).
    parameter fpnew_pkg::ifmt_logic_t  IntFmtConfig  = '1,
    // Parameter to enable vector operations (default: enabled).
    parameter logic                    EnableVectors = 1'b1,
    // Parameter to indicate the use of a PULP-based DivSqrt unit (default: enabled).
    parameter logic                    PulpDivsqrt   = 1'b1,
    // Parameter for the number of pipeline registers (default: 0).
    parameter int unsigned             NumPipeRegs   = 0,
    // Parameter for the pipeline configuration (default: BEFORE).
    parameter fpnew_pkg::pipe_config_t PipeConfig    = fpnew_pkg::BEFORE,
    // Parameter to enable external register enable signals (default: disabled).
    parameter logic                    ExtRegEna     = 1'b0,
    // Parameter for the tag type (default: logic).
    parameter type                     TagType       = logic,

    // Local parameter for the number of operands based on the operation group.
    localparam int unsigned NumOperands = fpnew_pkg::num_operands(OpGroup),
    // Local parameter for the number of supported floating-point formats.
    localparam int unsigned NumFormats = fpnew_pkg::NUM_FP_FORMATS,
    // Local parameter for the number of SIMD lanes.
    localparam int unsigned NumSimdLanes = fpnew_pkg::max_num_lanes(
        Width, FpFmtConfig, EnableVectors
    ),
    // Local parameter for the mask type based on the number of SIMD lanes.
    localparam type MaskType = logic [NumSimdLanes-1:0],
    // Local parameter for the width of the external register enable signal.
    localparam int unsigned ExtRegEnaWidth = NumPipeRegs == 0 ? 1 : NumPipeRegs
) (
    // Clock input.
    input logic clk_i,
    // Asynchronous reset input (active low).
    input logic rst_ni,

    // Input operands array.
    input logic                   [NumOperands-1:0][      Width-1:0] operands_i,
    // Input indicating if operands are boxed (per format and operand).
    input logic                   [ NumFormats-1:0][NumOperands-1:0] is_boxed_i,
    // Input rounding mode.
    input fpnew_pkg::roundmode_e                                     rnd_mode_i,
    // Input operation type.
    input fpnew_pkg::operation_e                                     op_i,
    // Input operation modifier.
    input logic                                                      op_mod_i,
    // Input source floating-point format.
    input fpnew_pkg::fp_format_e                                     src_fmt_i,
    // Input destination floating-point format.
    input fpnew_pkg::fp_format_e                                     dst_fmt_i,
    // Input integer format.
    input fpnew_pkg::int_format_e                                    int_fmt_i,
    // Input indicating if the operation is vectorial.
    input logic                                                      vectorial_op_i,
    // Input tag value.
    input TagType                                                    tag_i,
    // Input SIMD mask.
    input MaskType                                                   simd_mask_i,

    // Input validity signal.
    input  logic in_valid_i,
    // Output ready signal.
    output logic in_ready_o,
    // Input flush signal.
    input  logic flush_i,

    // Output result.
    output logic               [Width-1:0] result_o,
    // Output status flags.
    output fpnew_pkg::status_t             status_o,
    // Output extension bit.
    output logic                           extension_bit_o,
    // Output tag value.
    output TagType                         tag_o,

    // Output validity signal.
    output logic out_valid_o,
    // Input ready signal.
    input  logic out_ready_i,

    // Output busy signal.
    output logic busy_o,

    // Input register enable signals.
    input logic [ExtRegEnaWidth-1:0] reg_ena_i
);

  // Assert if DivSqrt is selected but not supported in the configuration.
  if ((OpGroup == fpnew_pkg::DIVSQRT) && !PulpDivsqrt &&
      !((FpFmtConfig[0] == 1) && (FpFmtConfig[1:NumFormats-1] == '0))) begin : g_assert_divsqrt
    $fatal(1, "T-Head-based DivSqrt unit supported only in FP32-only configurations");
  end

  // Local parameter for the maximum floating-point width.
  localparam int unsigned MaxFpWidth = fpnew_pkg::max_fp_width(FpFmtConfig);
  // Local parameter for the maximum integer width.
  localparam int unsigned MaxIntWidth = fpnew_pkg::max_int_width(IntFmtConfig);
  // Local parameter for the number of lanes considering all formats.
  localparam int unsigned NumLanes = fpnew_pkg::max_num_lanes(Width, FpFmtConfig, 1'b1);
  // Local parameter for the number of supported integer formats.
  localparam int unsigned NumIntFormats = fpnew_pkg::NUM_INT_FORMATS;

  // Local parameter for the number of bits needed to represent format.
  localparam int unsigned FmtBits = fpnew_pkg::maximum($clog2(NumFormats), $clog2(NumIntFormats));
  // Local parameter for the number of auxiliary data bits.
  localparam int unsigned AuxBits = FmtBits + 2;

  // Signals for lane ready and valid, and DivSqrt done and ready.
  logic [NumLanes-1:0] lane_in_ready, lane_out_valid, divsqrt_done, divsqrt_ready;
  // Signal indicating a vectorial operation.
  logic               vectorial_op;
  // Signal for the destination format.
  logic [FmtBits-1:0] dst_fmt;
  // Signal for auxiliary data.
  logic [AuxBits-1:0] aux_data;


  // Signals indicating if the destination format is integer or compact.
  logic dst_fmt_is_int, dst_is_cpk;
  // Signal for destination vector operation mode.
  logic [1:0] dst_vec_op;
  // Signal for target auxiliary data.
  logic [2:0] target_aux_d;
  // Signals indicating upcast or downcast operations.
  logic is_up_cast, is_down_cast;

  // Signals to store the result of each format slice.
  logic [NumFormats-1:0][Width-1:0] fmt_slice_result;
  // Signals to store the integer result of each format slice.
  logic [NumIntFormats-1:0][Width-1:0] ifmt_slice_result;

  // Signals for the conversion target data and its registered version.
  logic [Width-1:0] conv_target_d, conv_target_q;

  // Arrays to store lane-wise status, extension bit, tag, mask, auxiliary data, and busy status.
  fpnew_pkg::status_t [NumLanes-1:0]              lane_status;
  logic               [NumLanes-1:0]              lane_ext_bit;
  TagType             [NumLanes-1:0]              lane_tags;
  logic               [NumLanes-1:0]              lane_masks;
  logic               [NumLanes-1:0][AuxBits-1:0] lane_aux;
  logic               [NumLanes-1:0]              lane_busy;

  // Signals related to the result format and vector operation.
  logic                                           result_is_vector;
  logic               [ FmtBits-1:0]              result_fmt;
  logic result_fmt_is_int, result_is_cpk;
  logic [1:0] result_vec_op;

  // Signals for SIMD synchronization ready and done.
  logic simd_synch_rdy, simd_synch_done;




  // Assign the overall input ready based on the first lane.
  assign in_ready_o = lane_in_ready[0];
  // Enable vector operations if both input and parameter allow it.
  assign vectorial_op = vectorial_op_i & EnableVectors;


  // Determine if the destination format is integer based on operation group and type.
  assign dst_fmt_is_int = (OpGroup == fpnew_pkg::CONV) & (op_i == fpnew_pkg::F2I);
  // Determine if the destination is a compact packed format.
  assign dst_is_cpk     = (OpGroup == fpnew_pkg::CONV) & (op_i == fpnew_pkg::CPKAB ||
                                                              op_i == fpnew_pkg::CPKCD);
  // Determine the destination vector operation mode for compact packing.
  assign dst_vec_op = (OpGroup == fpnew_pkg::CONV) & {(op_i == fpnew_pkg::CPKCD), op_mod_i};

  // Determine if the conversion is an upcast or downcast.
  assign is_up_cast = (fpnew_pkg::fp_width(dst_fmt_i) > fpnew_pkg::fp_width(src_fmt_i));
  assign is_down_cast = (fpnew_pkg::fp_width(dst_fmt_i) < fpnew_pkg::fp_width(src_fmt_i));


  // Select the destination format (integer or floating-point).
  assign dst_fmt = dst_fmt_is_int ? int_fmt_i : dst_fmt_i;


  // Combine auxiliary data.
  assign aux_data = {dst_fmt_is_int, vectorial_op, dst_fmt};
  // Combine target auxiliary data.
  assign target_aux_d = {dst_vec_op, dst_is_cpk};


  // Determine the conversion target operand based on the operation group.
  if (OpGroup == fpnew_pkg::CONV) begin : g_conv_target
    assign conv_target_d = dst_is_cpk ? operands_i[2] : operands_i[1];
  end else begin : g_not_conv_target
    assign conv_target_d = '0;
  end


  // Signals to store if operands are boxed for different numbers of operands.
  logic [NumFormats-1:0]    is_boxed_1op;
  logic [NumFormats-1:0][1:0] is_boxed_2op;

  // Combine the is_boxed inputs for easier access.
  always_comb begin : boxed_2op
    for (int fmt = 0; fmt < NumFormats; fmt++) begin
      is_boxed_1op[fmt] = is_boxed_i[fmt][0];
      is_boxed_2op[fmt] = is_boxed_i[fmt][1:0];
    end
  end




  // Generate logic for each SIMD lane.
  for (genvar lane = 0; lane < int'(NumLanes); lane++) begin : gen_num_lanes
    // Local parameter for the current lane index.
    localparam int unsigned LANE = unsigned'(lane);

    // Local parameter for the active floating-point formats in this lane.
    localparam fpnew_pkg::fmt_logic_t ActiveFormats = fpnew_pkg::get_lane_formats(
        Width, FpFmtConfig, LANE
    );
    // Local parameter for the active integer formats in this lane.
    localparam fpnew_pkg::ifmt_logic_t ActiveIntFormats = fpnew_pkg::get_lane_int_formats(
        Width, FpFmtConfig, IntFmtConfig, LANE
    );
    // Local parameter for the maximum width of the active floating-point formats.
    localparam int unsigned MaxWidth = fpnew_pkg::max_fp_width(ActiveFormats);


    // Local parameter for the conversion-specific floating-point formats in this lane.
    localparam fpnew_pkg::fmt_logic_t ConvFormats = fpnew_pkg::get_conv_lane_formats(
        Width, FpFmtConfig, LANE
    );
    // Local parameter for the conversion-specific integer formats in this lane.
    localparam fpnew_pkg::ifmt_logic_t ConvIntFormats = fpnew_pkg::get_conv_lane_int_formats(
        Width, FpFmtConfig, IntFmtConfig, LANE
    );
    // Local parameter for the maximum width of the conversion formats.
    localparam int unsigned ConvWidth = fpnew_pkg::max_fp_width(ConvFormats);


    // Local parameter to select between active and conversion formats.
    localparam fpnew_pkg::fmt_logic_t LaneFormats = (OpGroup == fpnew_pkg::CONV)
                                                               ? ConvFormats : ActiveFormats;
    // Local parameter to select between maximum and conversion widths.
    localparam int unsigned LaneWidth = (OpGroup == fpnew_pkg::CONV) ? ConvWidth : MaxWidth;

    // Local signal to store the result of the current lane.
    logic [LaneWidth-1:0] local_result;


    // Instantiate logic only for the first lane or if vector operations are enabled.
    if ((lane == 0) || EnableVectors) begin : g_active_lane
      // Local signals for input/output validity and ready.
      logic in_valid, out_valid, out_ready;

      // Local signals for operands and the operation result and status.
      logic               [NumOperands-1:0][LaneWidth-1:0] local_operands;
      logic               [  LaneWidth-1:0]                op_result;
      fpnew_pkg::status_t                                  op_status;

      // Lane is valid if the overall input is valid and it's the first lane or a vector op.
      assign in_valid = in_valid_i & ((lane == 0) | vectorial_op);


      // Prepare the input operands for the current lane.
      always_comb begin : prepare_input
        for (int unsigned i = 0; i < NumOperands; i++) begin
          if (i == 2) begin
            // Third operand is shifted based on destination format width.
            local_operands[i] = operands_i[i] >> LANE * fpnew_pkg::fp_width(dst_fmt_i);
          end else begin
            // Other operands are shifted based on source format width.
            local_operands[i] = operands_i[i] >> LANE * fpnew_pkg::fp_width(src_fmt_i);
          end
        end


        // Adjust operands based on the specific conversion operation.
        if (OpGroup == fpnew_pkg::CONV) begin

          // For integer to float conversion, shift based on integer format width.
          if (op_i == fpnew_pkg::I2F) begin
            local_operands[0] = operands_i[0] >> LANE * fpnew_pkg::int_width(int_fmt_i);

            // For float to float conversion, handle potential upcasting in vector mode.
          end else if (op_i == fpnew_pkg::F2F) begin
            if (vectorial_op && op_mod_i && is_up_cast) begin
              local_operands[0] = operands_i[0] >>
                  LANE * fpnew_pkg::fp_width(src_fmt_i) + MaxFpWidth / 2;
            end

            // For compact packing operations, handle the second lane.
          end else if (dst_is_cpk) begin
            if (lane == 1) begin
              local_operands[0] = operands_i[1][LaneWidth-1:0];
            end
          end
        end
      end


      // Instantiate the floating-point multiply-add unit.
      if (OpGroup == fpnew_pkg::ADDMUL) begin : g_lane_instance
        fpnew_fma_multi #(
            .FpFmtConfig(LaneFormats),
            .NumPipeRegs(NumPipeRegs),
            .PipeConfig (PipeConfig),
            .TagType    (TagType),
            .AuxType    (logic [AuxBits-1:0])
        ) i_fpnew_fma_multi (
            .clk_i,
            .rst_ni,
            .operands_i     (local_operands),
            .is_boxed_i,
            .rnd_mode_i,
            .op_i,
            .op_mod_i,
            .src_fmt_i,
            .dst_fmt_i,
            .tag_i,
            .mask_i         (simd_mask_i[lane]),
            .aux_i          (aux_data),
            .in_valid_i     (in_valid),
            .in_ready_o     (lane_in_ready[lane]),
            .flush_i,
            .result_o       (op_result),
            .status_o       (op_status),
            .extension_bit_o(lane_ext_bit[lane]),
            .tag_o          (lane_tags[lane]),
            .mask_o         (lane_masks[lane]),
            .aux_o          (lane_aux[lane]),
            .out_valid_o    (out_valid),
            .out_ready_i    (out_ready),
            .busy_o         (lane_busy[lane]),
            .reg_ena_i
        );

        // Instantiate the floating-point divide/square root unit.
      end else if (OpGroup == fpnew_pkg::DIVSQRT) begin : g_lane_instance
        // Use the T-Head optimized version for FP32-only configurations.
        if (!PulpDivsqrt && LaneFormats[0]
          && (LaneFormats[1:fpnew_pkg::NUM_FP_FORMATS-1] == '0)) begin : g_lane_instance

          fpnew_divsqrt_th_32 #(
              .NumPipeRegs(NumPipeRegs),
              .PipeConfig (PipeConfig),
              .TagType    (TagType),
              .AuxType    (logic [AuxBits-1:0])
          ) i_fpnew_divsqrt_multi_th (
              .clk_i,
              .rst_ni,
              .operands_i     (local_operands[1:0]),
              .is_boxed_i     (is_boxed_2op),
              .rnd_mode_i,
              .op_i,
              .tag_i,
              .mask_i         (simd_mask_i[lane]),
              .aux_i          (aux_data),
              .in_valid_i     (in_valid),
              .in_ready_o     (lane_in_ready[lane]),
              .flush_i,
              .result_o       (op_result),
              .status_o       (op_status),
              .extension_bit_o(lane_ext_bit[lane]),
              .tag_o          (lane_tags[lane]),
              .mask_o         (lane_masks[lane]),
              .aux_o          (lane_aux[lane]),
              .out_valid_o    (out_valid),
              .out_ready_i    (out_ready),
              .busy_o         (lane_busy[lane]),
              .reg_ena_i
          );
          // Use the standard divide/square root unit.
        end else begin : g_lane_instance
          fpnew_divsqrt_multi #(
              .FpFmtConfig(LaneFormats),
              .NumPipeRegs(NumPipeRegs),
              .PipeConfig (PipeConfig),
              .TagType    (TagType),
              .AuxType    (logic [AuxBits-1:0])
          ) i_fpnew_divsqrt_multi (
              .clk_i,
              .rst_ni,
              .operands_i       (local_operands[1:0]),
              .is_boxed_i       (is_boxed_2op),
              .rnd_mode_i,
              .op_i,
              .dst_fmt_i,
              .tag_i,
              .mask_i           (simd_mask_i[lane]),
              .aux_i            (aux_data),
              .vectorial_op_i   (vectorial_op),
              .in_valid_i       (in_valid),
              .in_ready_o       (lane_in_ready[lane]),
              .divsqrt_done_o   (divsqrt_done[lane]),
              .simd_synch_done_i(simd_synch_done),
              .divsqrt_ready_o  (divsqrt_ready[lane]),
              .simd_synch_rdy_i (simd_synch_rdy),
              .flush_i,
              .result_o         (op_result),
              .status_o         (op_status),
              .extension_bit_o  (lane_ext_bit[lane]),
              .tag_o            (lane_tags[lane]),
              .mask_o           (lane_masks[lane]),
              .aux_o            (lane_aux[lane]),
              .out_valid_o      (out_valid),
              .out_ready_i      (out_ready),
              .busy_o           (lane_busy[lane]),
              .reg_ena_i
          );
        end
        // Placeholder for non-computational operations.
      end else if (OpGroup == fpnew_pkg::NONCOMP) begin : g_lane_instance

        // Instantiate the floating-point cast unit.
      end else if (OpGroup == fpnew_pkg::CONV) begin : g_lane_instance
        fpnew_cast_multi #(
            .FpFmtConfig (LaneFormats),
            .IntFmtConfig(ConvIntFormats),
            .NumPipeRegs (NumPipeRegs),
            .PipeConfig  (PipeConfig),
            .TagType     (TagType),
            .AuxType     (logic [AuxBits-1:0])
        ) i_fpnew_cast_multi (
            .clk_i,
            .rst_ni,
            .operands_i     (local_operands[0]),
            .is_boxed_i     (is_boxed_1op),
            .rnd_mode_i,
            .op_i,
            .op_mod_i,
            .src_fmt_i,
            .dst_fmt_i,
            .int_fmt_i,
            .tag_i,
            .mask_i         (simd_mask_i[lane]),
            .aux_i          (aux_data),
            .in_valid_i     (in_valid),
            .in_ready_o     (lane_in_ready[lane]),
            .flush_i,
            .result_o       (op_result),
            .status_o       (op_status),
            .extension_bit_o(lane_ext_bit[lane]),
            .tag_o          (lane_tags[lane]),
            .mask_o         (lane_masks[lane]),
            .aux_o          (lane_aux[lane]),
            .out_valid_o    (out_valid),
            .out_ready_i    (out_ready),
            .busy_o         (lane_busy[lane]),
            .reg_ena_i
        );
      end


      // Lane output is ready if the overall output is ready and it's the first lane or vector op.
      assign out_ready = out_ready_i & ((lane == 0) | result_is_vector);
      // Lane output is valid if the internal operation is valid and it's the first or vector op.
      assign lane_out_valid[lane] = out_valid & ((lane == 0) | result_is_vector);


      // Assign the local result, considering external register enable.
      assign local_result = (lane_out_valid[lane] | ExtRegEna) ? op_result : '{
              default: lane_ext_bit[0]
          };
      // Assign the lane status, considering external register enable.
      assign lane_status[lane] = (lane_out_valid[lane] | ExtRegEna) ? op_status : '0;


      // If not the first lane and vector ops are disabled, keep outputs inactive.
    end else begin : g_inactive_lane
      assign lane_out_valid[lane] = 1'b0;
      assign lane_in_ready[lane]  = 1'b0;
      assign lane_aux[lane]       = 1'b0;
      assign lane_masks[lane]     = 1'b1;
      assign lane_tags[lane]      = 1'b0;
      assign divsqrt_done[lane]   = 1'b0;
      assign divsqrt_ready[lane]  = 1'b0;
      assign lane_ext_bit[lane]   = 1'b1;
      assign local_result         = {(LaneWidth) {lane_ext_bit[0]}};
      assign lane_status[lane]    = '0;
      assign lane_busy[lane]      = 1'b0;
    end


    // Pack the floating-point result for the current lane.
    for (genvar fmt = 0; fmt < NumFormats; fmt++) begin : g_pack_fp_result

      // Local parameter for the width of the current floating-point format.
      localparam int unsigned FpWidth = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));

      // Assign the result if the format is active in this lane.
      if (ActiveFormats[fmt]) begin : g_pack_fp_result_active
        assign fmt_slice_result[fmt][(LANE+1)*FpWidth-1:LANE*FpWidth] = local_result[FpWidth-1:0];
        // Extend with the lane's extension bit if within the total width.
      end else if ((LANE + 1) * FpWidth <= Width) begin : g_extend_fp_result
        assign fmt_slice_result[fmt][(LANE+1)*FpWidth-1:LANE*FpWidth] = '{
                default: lane_ext_bit[LANE]
            };
        // Extend with the lane's extension bit if partially within the total width.
      end else if (LANE * FpWidth < Width) begin : g_extend_fp_result
        assign fmt_slice_result[fmt][Width-1:LANE*FpWidth] = '{default: lane_ext_bit[LANE]};
      end
    end


    // Pack the integer result for the current lane if it's a conversion operation.
    if (OpGroup == fpnew_pkg::CONV) begin : g_int_results_enabled
      for (genvar ifmt = 0; ifmt < NumIntFormats; ifmt++) begin : g_pack_int_result

        // Local parameter for the width of the current integer format.
        localparam int unsigned IntWidth = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));
        // Assign the result if the integer format is active.
        if (ActiveIntFormats[ifmt]) begin : g_pack_int_result_active
          assign ifmt_slice_result[ifmt][(LANE+1)*IntWidth-1:LANE*IntWidth] =
              local_result[IntWidth-1:0];
          // Pad with zeros if within the total width.
        end else if ((LANE + 1) * IntWidth <= Width) begin : g_pack_int_result_pad
          assign ifmt_slice_result[ifmt][(LANE+1)*IntWidth-1:LANE*IntWidth] = '0;
          // Pad with zeros if partially within the total width.
        end else if (LANE * IntWidth < Width) begin : g_pack_int_result_pad
          assign ifmt_slice_result[ifmt][Width-1:LANE*IntWidth] = '0;
        end
      end
    end
  end


  // Extend the floating-point result to the full width.
  for (genvar fmt = 0; fmt < NumFormats; fmt++) begin : g_extend_fp_result

    // Local parameter for the width of the current floating-point format.
    localparam int unsigned FpWidth = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    // Extend with the extension bit of the first lane if needed.
    if (NumLanes * FpWidth < Width)
      assign fmt_slice_result[fmt][Width-1:NumLanes*FpWidth] = '{default: lane_ext_bit[0]};
  end

  // Extend or mute the integer result based on the operation group.
  for (genvar ifmt = 0; ifmt < NumIntFormats; ifmt++) begin : g_extend_or_mute_int_result

    // Mute the integer result if it's not a conversion operation.
    if (OpGroup != fpnew_pkg::CONV) begin : g_mute_int_result
      assign ifmt_slice_result[ifmt] = '0;


      // Extend the integer result to the full width if it's a conversion.
    end else begin : g_extend_int_result

      // Local parameter for the width of the current integer format.
      localparam int unsigned IntWidth = fpnew_pkg::int_width(fpnew_pkg::int_format_e'(ifmt));
      // Pad with zeros if needed.
      if (NumLanes * IntWidth < Width)
        assign ifmt_slice_result[ifmt][Width-1:NumLanes*IntWidth] = '0;
    end
  end


  // Handle target register for conversion operations.
  if (OpGroup == fpnew_pkg::CONV) begin : target_regs

    // Pipeline registers for the target operand and auxiliary data.
    logic [0:NumPipeRegs][Width-1:0] byp_pipe_target_q;
    logic [0:NumPipeRegs][      2:0] byp_pipe_aux_q;
    logic [0:NumPipeRegs]            byp_pipe_valid_q;

    // Ready signals for the pipeline stages.
    logic [0:NumPipeRegs]            byp_pipe_ready;


    // Assign the first stage of the pipeline.
    assign byp_pipe_target_q[0] = conv_target_d;
    assign byp_pipe_aux_q[0] = target_aux_d;
    assign byp_pipe_valid_q[0] = in_valid_i & vectorial_op;

    // Generate the bypass pipeline.
    for (genvar i = 0; i < NumPipeRegs; i++) begin : gen_bypass_pipeline

      // Register enable signal for the current stage.
      logic reg_ena;



      // Ready signal for the current stage.
      assign byp_pipe_ready[i] = byp_pipe_ready[i+1] | ~byp_pipe_valid_q[i+1];

      // Register for the valid signal.
      always_ff @(posedge (clk_i) or negedge (rst_ni)) begin
        if (!rst_ni) begin
          byp_pipe_valid_q[i+1] <= '0;
        end else begin
          byp_pipe_valid_q[i+1] <= (flush_i) ? '0 : (byp_pipe_ready[i]) ? (byp_pipe_valid_q[i]) : (byp_pipe_valid_q[i+1]);
        end
      end

      // Determine the register enable signal.
      assign reg_ena = (byp_pipe_ready[i] & byp_pipe_valid_q[i]) | reg_ena_i[i];

      // Registers for the target operand and auxiliary data.
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          byp_pipe_target_q[i+1] <= '0;
        end else begin
          byp_pipe_target_q[i+1] <= (reg_ena) ? (byp_pipe_target_q[i]) : (byp_pipe_target_q[i+1]);
        end
      end

      // Register for the auxiliary data.
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          byp_pipe_aux_q[i+1] <= '0;
        end else begin
          byp_pipe_aux_q[i+1] <= (reg_ena) ? (byp_pipe_aux_q[i]) : (byp_pipe_aux_q[i+1]);
        end
      end
    end

    // Ready signal for the last stage.
    assign byp_pipe_ready[NumPipeRegs] = out_ready_i & result_is_vector;

    // Output of the pipeline.
    assign conv_target_q = byp_pipe_target_q[NumPipeRegs];


    // Extract result vector operation mode and compact packing flag.
    assign {result_vec_op, result_is_cpk} = byp_pipe_aux_q[NumPipeRegs];
  end else begin : g_no_conv
    // Default values if not a conversion operation.
    assign {result_vec_op, result_is_cpk} = '0;
    assign conv_target_q = '0;
  end

  // Handle SIMD synchronization for PulpDivsqrt unit.
  if (PulpDivsqrt) begin : g_pulp_divsqrt_sync

    // All lanes must be ready/done if vector operation is enabled.
    assign simd_synch_rdy  = EnableVectors ? &divsqrt_ready : divsqrt_ready[0];
    assign simd_synch_done = EnableVectors ? &divsqrt_done : divsqrt_done[0];
  end else begin : g_no_pulp_divsqrt_sync

    // No synchronization needed if not using PulpDivsqrt.
    assign simd_synch_rdy  = '0;
    assign simd_synch_done = '0;
  end




  // Extract result format information from the first lane's auxiliary data.
  assign {result_fmt_is_int, result_is_vector, result_fmt} = lane_aux[0];

  // Select the result based on whether it's an integer or floating-point format.
  assign result_o        = result_fmt_is_int
                              ? ifmt_slice_result[result_fmt]
                              : fmt_slice_result[result_fmt];

  // Assign the extension bit, tag, and busy status from the first lane.
  assign extension_bit_o = lane_ext_bit[0];
  assign tag_o = lane_tags[0];
  assign busy_o = (|lane_busy);

  // Assign the output validity based on the first lane.
  assign out_valid_o = lane_out_valid[0];


  // Combine the status flags from all active lanes.
  always_comb begin : output_processing

    automatic fpnew_pkg::status_t temp_status;
    temp_status = '0;
    for (int i = 0; i < int'(NumLanes); i++) temp_status |= lane_status[i] & {5{lane_masks[i]}};
    status_o = temp_status;
  end

endmodule
