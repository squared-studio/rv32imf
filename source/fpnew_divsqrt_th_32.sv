module fpnew_divsqrt_th_32 #(
    // Number of pipeline registers
    parameter int unsigned             NumPipeRegs = 0,
    // Pipeline configuration (BEFORE, AFTER, DISTRIBUTED, etc.)
    parameter fpnew_pkg::pipe_config_t PipeConfig  = fpnew_pkg::BEFORE,
    // Type for tagging operations
    parameter type                     TagType     = logic,
    // Type for auxiliary data
    parameter type                     AuxType     = logic,

    // Local parameter for data width
    localparam int unsigned WIDTH          = 32,
    // Number of floating-point formats supported
    localparam int unsigned NUM_FORMATS    = fpnew_pkg::NUM_FP_FORMATS,
    // Width of the enable signal for pipeline registers
    localparam int unsigned ExtRegEnaWidth = NumPipeRegs == 0 ? 1 : NumPipeRegs
) (
    // Clock signal
    input logic clk_i,
    // Active-low reset signal
    input logic rst_ni,

    // Input operands (two 32-bit values)
    input logic                  [            1:0][WIDTH-1:0] operands_i,
    // Indicates if the operands are boxed
    input logic                  [NUM_FORMATS-1:0][      1:0] is_boxed_i,
    // Rounding mode for the operation
    input fpnew_pkg::roundmode_e                              rnd_mode_i,
    // Operation type (e.g., DIV, SQRT)
    input fpnew_pkg::operation_e                              op_i,
    // Tag associated with the operation
    input TagType                                             tag_i,
    // Mask signal for the operation
    input logic                                               mask_i,
    // Auxiliary data for the operation
    input AuxType                                             aux_i,

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

  // Number of input pipeline registers based on configuration
  localparam NUM_INP_REGS = (PipeConfig == fpnew_pkg::BEFORE)
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? (NumPipeRegs / 2)
                               : 0);
  // Number of output pipeline registers based on configuration
  localparam NUM_OUT_REGS = (PipeConfig == fpnew_pkg::AFTER || PipeConfig == fpnew_pkg::INSIDE)
                            ? NumPipeRegs
                            : (PipeConfig == fpnew_pkg::DISTRIBUTED
                               ? ((NumPipeRegs + 1) / 2)
                               : 0);

  // Input pipeline registers for operands, rounding mode, operation, etc.
  logic                  [           1:0][WIDTH-1:0]            operands_q;
  fpnew_pkg::roundmode_e                                        rnd_mode_q;
  fpnew_pkg::operation_e                                        op_q;
  logic                                                         in_valid_q;

  // Input pipeline registers for operands, rounding mode, operation, etc.
  logic                  [0:NUM_INP_REGS][      1:0][WIDTH-1:0] inp_pipe_operands_q;
  fpnew_pkg::roundmode_e [0:NUM_INP_REGS]                       inp_pipe_rnd_mode_q;
  fpnew_pkg::operation_e [0:NUM_INP_REGS]                       inp_pipe_op_q;
  TagType                [0:NUM_INP_REGS]                       inp_pipe_tag_q;
  logic                  [0:NUM_INP_REGS]                       inp_pipe_mask_q;
  AuxType                [0:NUM_INP_REGS]                       inp_pipe_aux_q;
  logic                  [0:NUM_INP_REGS]                       inp_pipe_valid_q;

  logic                  [0:NUM_INP_REGS]                       inp_pipe_ready;

  assign inp_pipe_operands_q[0] = operands_i;
  assign inp_pipe_rnd_mode_q[0] = rnd_mode_i;
  assign inp_pipe_op_q[0]       = op_i;
  assign inp_pipe_tag_q[0]      = tag_i;
  assign inp_pipe_mask_q[0]     = mask_i;
  assign inp_pipe_aux_q[0]      = aux_i;
  assign inp_pipe_valid_q[0]    = in_valid_i;

  assign in_ready_o             = inp_pipe_ready[0];

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
  assign rnd_mode_q = inp_pipe_rnd_mode_q[NUM_INP_REGS];
  assign op_q       = inp_pipe_op_q[NUM_INP_REGS];
  assign in_valid_q = inp_pipe_valid_q[NUM_INP_REGS];

  // FSM states for controlling the operation
  typedef enum logic [1:0] {
    IDLE,
    BUSY,
    HOLD
  } fsm_state_e;
  fsm_state_e state_q, state_d;

  logic in_ready;
  logic div_op, sqrt_op;
  logic unit_ready_q, unit_done;
  logic op_starting;
  logic out_valid, out_ready;
  logic hold_result;
  logic data_is_held;
  logic unit_busy;

  assign div_op = in_valid_q & (op_q == fpnew_pkg::DIV) & in_ready & ~flush_i;
  assign sqrt_op = in_valid_q & (op_q == fpnew_pkg::SQRT) & in_ready & ~flush_i;
  assign op_starting = div_op | sqrt_op;

  logic fdsu_fpu_ex1_stall, fdsu_fpu_ex1_stall_q;
  logic div_op_d, div_op_q;
  logic sqrt_op_d, sqrt_op_q;

  assign div_op_d  = (fdsu_fpu_ex1_stall) ? div_op : 1'b0;
  assign sqrt_op_d = (fdsu_fpu_ex1_stall) ? sqrt_op : 1'b0;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fdsu_fpu_ex1_stall_q <= ('0);
    end else begin
      fdsu_fpu_ex1_stall_q <= (1'b1) ? (fdsu_fpu_ex1_stall) : (fdsu_fpu_ex1_stall_q);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      div_op_q <= ('0);
    end else begin
      div_op_q <= (1'b1) ? (div_op_d) : (div_op_q);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sqrt_op_q <= ('0);
    end else begin
      sqrt_op_q <= (1'b1) ? (sqrt_op_d) : (sqrt_op_q);
    end
  end

  // FSM logic to control input readiness, output validity, and state transitions
  always_comb begin : flag_fsm

    in_ready                     = 1'b0;
    out_valid                    = 1'b0;
    hold_result                  = 1'b0;
    data_is_held                 = 1'b0;
    unit_busy                    = 1'b0;
    state_d                      = state_q;
    inp_pipe_ready[NUM_INP_REGS] = unit_ready_q;

    unique case (state_q)

      IDLE: begin

        in_ready = unit_ready_q;
        if (in_valid_q && unit_ready_q) begin
          inp_pipe_ready[NUM_INP_REGS] = unit_ready_q && !fdsu_fpu_ex1_stall;
          state_d = BUSY;
        end
      end

      BUSY: begin
        inp_pipe_ready[NUM_INP_REGS] = fdsu_fpu_ex1_stall_q;
        unit_busy = 1'b1;

        if (unit_done) begin
          out_valid = 1'b1;

          if (out_ready) begin
            state_d = IDLE;
            if (in_valid_q && unit_ready_q) begin
              in_ready = 1'b1;
              state_d  = BUSY;
            end

          end else begin
            hold_result = 1'b1;
            state_d     = HOLD;
          end
        end
      end

      HOLD: begin
        unit_busy    = 1'b1;
        data_is_held = 1'b1;
        out_valid    = 1'b1;

        if (out_ready) begin
          state_d = IDLE;
          if (in_valid_q && unit_ready_q) begin
            in_ready = 1'b1;
            state_d  = BUSY;
          end
        end
      end

      default: state_d = IDLE;
    endcase

    if (flush_i) begin
      unit_busy = 1'b0;
      out_valid = 1'b0;
      state_d   = IDLE;
    end
  end


  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= (IDLE);
    end else begin
      state_q <= (state_d);
    end
  end


  TagType result_tag_q;
  AuxType result_aux_q;
  logic   result_mask_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      result_tag_q <= ('0);
    end else begin
      result_tag_q <= (op_starting) ? (inp_pipe_tag_q[NUM_INP_REGS]) : (result_tag_q);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      result_mask_q <= ('0);
    end else begin
      result_mask_q <= (op_starting) ? (inp_pipe_mask_q[NUM_INP_REGS]) : (result_mask_q);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      result_aux_q <= ('0);
    end else begin
      result_aux_q <= (op_starting) ? (inp_pipe_aux_q[NUM_INP_REGS]) : (result_aux_q);
    end
  end

  logic [WIDTH-1:0] unit_result, held_result_q;
  fpnew_pkg::status_t unit_status, held_status_q;

  logic        ctrl_fdsu_ex1_sel;
  logic        fdsu_fpu_ex1_cmplt;
  logic [ 4:0] fdsu_fpu_ex1_fflags;
  logic [ 7:0] fdsu_fpu_ex1_special_sel;
  logic [ 3:0] fdsu_fpu_ex1_special_sign;
  logic        fdsu_fpu_no_op;
  logic [ 2:0] idu_fpu_ex1_eu_sel;
  logic [31:0] fdsu_frbus_data;
  logic [ 4:0] fdsu_frbus_fflags;
  logic        fdsu_frbus_wb_vld;

  logic [31:0] dp_frbus_ex2_data;
  logic [ 4:0] dp_frbus_ex2_fflags;
  logic [ 2:0] dp_xx_ex1_cnan;
  logic [ 2:0] dp_xx_ex1_id;
  logic [ 2:0] dp_xx_ex1_inf;
  logic [ 2:0] dp_xx_ex1_norm;
  logic [ 2:0] dp_xx_ex1_qnan;
  logic [ 2:0] dp_xx_ex1_snan;
  logic [ 2:0] dp_xx_ex1_zero;
  logic        ex2_inst_wb;
  logic ex2_inst_wb_vld_d, ex2_inst_wb_vld_q;

  logic [31:0] fpu_idu_fwd_data;
  logic [ 4:0] fpu_idu_fwd_fflags;
  logic        fpu_idu_fwd_vld;

  logic        unit_ready_d;

  always_comb begin
    if (op_starting && unit_ready_q) begin
      if (ex2_inst_wb && ex2_inst_wb_vld_q) begin
        unit_ready_d = 1'b1;
      end else begin
        unit_ready_d = 1'b0;
      end
    end else if (fpu_idu_fwd_vld | flush_i) begin
      unit_ready_d = 1'b1;
    end else begin
      unit_ready_d = unit_ready_q;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      unit_ready_q <= (1'b1);
    end else begin
      unit_ready_q <= (1'b1) ? (unit_ready_d) : (unit_ready_q);
    end
  end

  always_comb begin
    ctrl_fdsu_ex1_sel  = 1'b0;
    idu_fpu_ex1_eu_sel = 3'h0;
    if (op_starting) begin
      ctrl_fdsu_ex1_sel  = 1'b1;
      idu_fpu_ex1_eu_sel = 3'h4;
    end else if (fdsu_fpu_ex1_stall_q) begin
      ctrl_fdsu_ex1_sel  = 1'b1;
      idu_fpu_ex1_eu_sel = 3'h4;
    end else begin
      ctrl_fdsu_ex1_sel  = 1'b0;
      idu_fpu_ex1_eu_sel = 3'h0;
    end
  end

  // Instantiate the floating-point division/square root unit
  pa_fdsu_top i_divsqrt_thead (
      .cp0_fpu_icg_en           (1'b0),
      .cp0_fpu_xx_dqnan         (1'b0),
      .cp0_yy_clk_en            (1'b1),
      .cpurst_b                 (rst_ni),
      .ctrl_fdsu_ex1_sel        (ctrl_fdsu_ex1_sel),
      .ctrl_xx_ex1_cmplt_dp     (ctrl_fdsu_ex1_sel),
      .ctrl_xx_ex1_inst_vld     (ctrl_fdsu_ex1_sel),
      .ctrl_xx_ex1_stall        (fdsu_fpu_ex1_stall),
      .ctrl_xx_ex1_warm_up      (1'b0),
      .ctrl_xx_ex2_warm_up      (1'b0),
      .ctrl_xx_ex3_warm_up      (1'b0),
      .dp_xx_ex1_cnan           (dp_xx_ex1_cnan),
      .dp_xx_ex1_id             (dp_xx_ex1_id),
      .dp_xx_ex1_inf            (dp_xx_ex1_inf),
      .dp_xx_ex1_qnan           (dp_xx_ex1_qnan),
      .dp_xx_ex1_rm             (rnd_mode_q),
      .dp_xx_ex1_snan           (dp_xx_ex1_snan),
      .dp_xx_ex1_zero           (dp_xx_ex1_zero),
      .fdsu_fpu_debug_info      (),
      .fdsu_fpu_ex1_cmplt       (fdsu_fpu_ex1_cmplt),
      .fdsu_fpu_ex1_cmplt_dp    (),
      .fdsu_fpu_ex1_fflags      (fdsu_fpu_ex1_fflags),
      .fdsu_fpu_ex1_special_sel (fdsu_fpu_ex1_special_sel),
      .fdsu_fpu_ex1_special_sign(fdsu_fpu_ex1_special_sign),
      .fdsu_fpu_ex1_stall       (fdsu_fpu_ex1_stall),
      .fdsu_fpu_no_op           (fdsu_fpu_no_op),
      .fdsu_frbus_data          (fdsu_frbus_data),
      .fdsu_frbus_fflags        (fdsu_frbus_fflags),
      .fdsu_frbus_freg          (),
      .fdsu_frbus_wb_vld        (fdsu_frbus_wb_vld),
      .forever_cpuclk           (clk_i),
      .frbus_fdsu_wb_grant      (fdsu_frbus_wb_vld),
      .idu_fpu_ex1_dst_freg     (5'h0f),
      .idu_fpu_ex1_eu_sel       (idu_fpu_ex1_eu_sel),
      .idu_fpu_ex1_func         ({8'b0, div_op | div_op_q, sqrt_op | sqrt_op_q}),
      .idu_fpu_ex1_srcf0        (operands_q[0][31:0]),
      .idu_fpu_ex1_srcf1        (operands_q[1][31:0]),
      .pad_yy_icg_scan_en       (1'b0),
      .rtu_xx_ex1_cancel        (1'b0),
      .rtu_xx_ex2_cancel        (1'b0),
      .rtu_yy_xx_async_flush    (flush_i),
      .rtu_yy_xx_flush          (1'b0)
  );

  // Instantiate the floating-point datapath
  pa_fpu_dp x_pa_fpu_dp (
      .cp0_fpu_icg_en           (1'b0),
      .cp0_fpu_xx_rm            (rnd_mode_q),
      .cp0_yy_clk_en            (1'b1),
      .ctrl_xx_ex1_inst_vld     (ctrl_fdsu_ex1_sel),
      .ctrl_xx_ex1_stall        (1'b0),
      .ctrl_xx_ex1_warm_up      (1'b0),
      .dp_frbus_ex2_data        (dp_frbus_ex2_data),
      .dp_frbus_ex2_fflags      (dp_frbus_ex2_fflags),
      .dp_xx_ex1_cnan           (dp_xx_ex1_cnan),
      .dp_xx_ex1_id             (dp_xx_ex1_id),
      .dp_xx_ex1_inf            (dp_xx_ex1_inf),
      .dp_xx_ex1_norm           (dp_xx_ex1_norm),
      .dp_xx_ex1_qnan           (dp_xx_ex1_qnan),
      .dp_xx_ex1_snan           (dp_xx_ex1_snan),
      .dp_xx_ex1_zero           (dp_xx_ex1_zero),
      .ex2_inst_wb              (ex2_inst_wb),
      .fdsu_fpu_ex1_fflags      (fdsu_fpu_ex1_fflags),
      .fdsu_fpu_ex1_special_sel (fdsu_fpu_ex1_special_sel),
      .fdsu_fpu_ex1_special_sign(fdsu_fpu_ex1_special_sign),
      .forever_cpuclk           (clk_i),
      .idu_fpu_ex1_eu_sel       (idu_fpu_ex1_eu_sel),
      .idu_fpu_ex1_func         ({8'b0, div_op, sqrt_op}),
      .idu_fpu_ex1_gateclk_vld  (fdsu_fpu_ex1_cmplt),
      .idu_fpu_ex1_rm           (rnd_mode_q),
      .idu_fpu_ex1_srcf0        (operands_q[0][31:0]),
      .idu_fpu_ex1_srcf1        (operands_q[1][31:0]),
      .idu_fpu_ex1_srcf2        ('0),
      .pad_yy_icg_scan_en       (1'b0)
  );

  assign ex2_inst_wb_vld_d = ctrl_fdsu_ex1_sel;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ex2_inst_wb_vld_q <= ('0);
    end else begin
      ex2_inst_wb_vld_q <= (ex2_inst_wb_vld_d);
    end
  end

  // Instantiate the floating-point result bus
  pa_fpu_frbus x_pa_fpu_frbus (
      .ctrl_frbus_ex2_wb_req(ex2_inst_wb & ex2_inst_wb_vld_q),
      .dp_frbus_ex2_data    (dp_frbus_ex2_data),
      .dp_frbus_ex2_fflags  (dp_frbus_ex2_fflags),
      .fdsu_frbus_data      (fdsu_frbus_data),
      .fdsu_frbus_fflags    (fdsu_frbus_fflags),
      .fdsu_frbus_wb_vld    (fdsu_frbus_wb_vld),
      .fpu_idu_fwd_data     (fpu_idu_fwd_data),
      .fpu_idu_fwd_fflags   (fpu_idu_fwd_fflags),
      .fpu_idu_fwd_vld      (fpu_idu_fwd_vld)
  );

  always_comb begin
    unit_result[31:0] = fpu_idu_fwd_data[31:0];
    unit_status[4:0]  = fpu_idu_fwd_fflags[4:0];
    unit_done         = fpu_idu_fwd_vld;
  end

  always_ff @(posedge (clk_i)) begin
    held_result_q <= (hold_result) ? (unit_result) : (held_result_q);
  end

  always_ff @(posedge (clk_i)) begin
    held_status_q <= (hold_result) ? (unit_status) : (held_status_q);
  end

  logic [WIDTH-1:0] result_d;
  fpnew_pkg::status_t status_d;

  assign result_d = data_is_held ? held_result_q : unit_result;
  assign status_d = data_is_held ? held_status_q : unit_status;

  logic               [0:NUM_OUT_REGS][WIDTH-1:0] out_pipe_result_q;
  fpnew_pkg::status_t [0:NUM_OUT_REGS]            out_pipe_status_q;
  TagType             [0:NUM_OUT_REGS]            out_pipe_tag_q;
  AuxType             [0:NUM_OUT_REGS]            out_pipe_aux_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_mask_q;
  logic               [0:NUM_OUT_REGS]            out_pipe_valid_q;

  logic               [0:NUM_OUT_REGS]            out_pipe_ready;

  assign out_pipe_result_q[0] = result_d;
  assign out_pipe_status_q[0] = status_d;
  assign out_pipe_tag_q[0]    = result_tag_q;
  assign out_pipe_mask_q[0]   = result_mask_q;
  assign out_pipe_aux_q[0]    = result_aux_q;
  assign out_pipe_valid_q[0]  = out_valid;

  assign out_ready = out_pipe_ready[0];

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
  assign extension_bit_o              = 1'b1;
  assign tag_o                        = out_pipe_tag_q[NUM_OUT_REGS];
  assign mask_o                       = out_pipe_mask_q[NUM_OUT_REGS];
  assign aux_o                        = out_pipe_aux_q[NUM_OUT_REGS];
  assign out_valid_o                  = out_pipe_valid_q[NUM_OUT_REGS];
  assign busy_o                       = (|{inp_pipe_valid_q, unit_busy, out_pipe_valid_q});
endmodule
