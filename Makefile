# Set the default goal to 'help'
.DEFAULT_GOAL := help

# Define the root directory
ROOT := $(shell echo "$(realpath .)")

# Define the grep command for warnings and errors
GREP_EW := grep -E "WARNING:|ERROR:|" --color=auto

# Define the test argument
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	TESTPLUSARGS = --testplusarg DEBUG
else
	TESTPLUSARGS = 
endif

# if build/test exists, then TEST=read build/test, else none
ifeq ($(wildcard build/test),)
	TEST ?=
else
	TEST := $(file <build/test)
endif

RISCV64_GCC ?= riscv64-unknown-elf-gcc
RISCV64_OBJCOPY ?= riscv64-unknown-elf-objcopy
RISCV64_NM ?= riscv64-unknown-elf-nm
RISCV64_OBJDUMP ?= riscv64-unknown-elf-objdump

################################################################################
# Add all the RTL source files to the LIB variable
################################################################################

LIB += ${ROOT}/source/rv32imf_pkg.sv
LIB += ${ROOT}/source/rv32imf_fpu_pkg.sv
LIB += ${ROOT}/source/fpnew_pkg.sv
LIB += ${ROOT}/source/rv32imf_clock_gate.sv
LIB += ${ROOT}/source/rv32imf_sleep_unit.sv
LIB += ${ROOT}/source/rv32imf_prefetch_controller.sv
LIB += ${ROOT}/source/rv32imf_fifo.sv
LIB += ${ROOT}/source/rv32imf_obi_interface.sv
LIB += ${ROOT}/source/rv32imf_prefetch_buffer.sv
LIB += ${ROOT}/source/rv32imf_aligner.sv
LIB += ${ROOT}/source/rv32imf_compressed_decoder.sv
LIB += ${ROOT}/source/rv32imf_if_stage.sv
LIB += ${ROOT}/source/rv32imf_register_file.sv
LIB += ${ROOT}/source/rv32imf_decoder.sv
LIB += ${ROOT}/source/rv32imf_controller.sv
LIB += ${ROOT}/source/rv32imf_int_controller.sv
LIB += ${ROOT}/source/rv32imf_id_stage.sv
LIB += ${ROOT}/source/rv32imf_popcnt.sv
LIB += ${ROOT}/source/rv32imf_ff_one.sv
LIB += ${ROOT}/source/rv32imf_alu_div.sv
LIB += ${ROOT}/source/rv32imf_alu.sv
LIB += ${ROOT}/source/rv32imf_mult.sv
LIB += ${ROOT}/source/rv32imf_apu_disp.sv
LIB += ${ROOT}/source/rv32imf_ex_stage.sv
LIB += ${ROOT}/source/rv32imf_load_store_unit.sv
LIB += ${ROOT}/source/rv32imf_cs_registers.sv
LIB += ${ROOT}/source/rv32imf_core.sv
LIB += ${ROOT}/source/fpnew_classifier.sv
LIB += ${ROOT}/source/lzc.sv
LIB += ${ROOT}/source/fpnew_rounding.sv
LIB += ${ROOT}/source/fpnew_fma_multi.sv
LIB += ${ROOT}/source/fpnew_opgroup_multifmt_slice.sv
LIB += ${ROOT}/source/rr_arb_tree.sv
LIB += ${ROOT}/source/fpnew_opgroup_block.sv
LIB += ${ROOT}/source/pa_fdsu_special.sv
LIB += ${ROOT}/source/pa_fdsu_ff1.sv
LIB += ${ROOT}/source/pa_fdsu_prepare.sv
LIB += ${ROOT}/source/gated_clk_cell.sv
LIB += ${ROOT}/source/pa_fdsu_srt_single.sv
LIB += ${ROOT}/source/pa_fdsu_round_single.sv
LIB += ${ROOT}/source/pa_fdsu_pack_single.sv
LIB += ${ROOT}/source/pa_fdsu_ctrl.sv
LIB += ${ROOT}/source/pa_fdsu_top.sv
LIB += ${ROOT}/source/pa_fpu_src_type.sv
LIB += ${ROOT}/source/pa_fpu_dp.sv
LIB += ${ROOT}/source/pa_fpu_frbus.sv
LIB += ${ROOT}/source/fpnew_divsqrt_th_32.sv
LIB += ${ROOT}/source/fpnew_noncomp.sv
LIB += ${ROOT}/source/fpnew_opgroup_fmt_slice.sv
LIB += ${ROOT}/source/fpnew_cast_multi.sv
LIB += ${ROOT}/source/fpnew_top.sv
LIB += ${ROOT}/source/rv32imf_fp_wrapper.sv
LIB += ${ROOT}/source/rv32imf.sv

################################################################################
# Add all the testbench files to the LIB variable
################################################################################

LIB += ${ROOT}/tb/sim_memory.sv
LIB += ${ROOT}/tb/rv32imf_tb.sv

################################################################################
# TARGETS
################################################################################

# Define the 'vivado' target to clean and run the build
.PHONY: vivado
vivado: clean run

# Define the 'clean' target to remove the build directory
.PHONY: clean
clean:
	@rm -rf build
	@make -s build

# Define the 'build' target to create the build directory and add it to gitignore
build:
	@mkdir -p build
	@echo "*" > build/.gitignore
	@git add build > /dev/null 2>&1

# Define the 'build/done' target to compile the project
build/done:
	@make -s compile

# Define the 'compile' target to compile the source files
.PHONY: compile
compile: build
	@cd build; xvlog -i ${ROOT}/include -sv ${LIB} -nolog | $(GREP_EW)
	@cd build; xelab rv32imf_tb -s top -nolog | $(GREP_EW)
	@echo "build done" > build/done

# Define the 'run' target to run the tests
.PHONY: run
run: build/done
	@echo -n "$(TEST)" > build/test
	@echo -e "\033[1;33mRunning $(TEST)\033[0m"
	@make -s test TEST=$(TEST)
	@cd build; xsim top $(TESTPLUSARGS) -runall -nolog | $(GREP_EW)
ifeq ($(DEBUG), 1)
	@make -s readable
endif

# Define the 'readable' target to make the trace file more readable
.PHONY: readable
readable: build/prog.trace
	@sed -i -e "s/GPR0:/x0\/zero:/g" \
		-e "s/GPR1:/x1\/ra:/g" \
		-e "s/GPR2:/x2\/sp:/g" \
		-e "s/GPR3:/x3\/gp:/g" \
		-e "s/GPR4:/x4\/tp:/g" \
		-e "s/GPR5:/x5\/t0:/g" \
		-e "s/GPR6:/x6\/t1:/g" \
		-e "s/GPR7:/x7\/t2:/g" \
		-e "s/GPR8:/x8\/s0\/fp:/g" \
		-e "s/GPR9:/x9\/s1:/g" \
		-e "s/GPR10:/x10\/a0:/g" \
		-e "s/GPR11:/x11\/a1:/g" \
		-e "s/GPR12:/x12\/a2:/g" \
		-e "s/GPR13:/x13\/a3:/g" \
		-e "s/GPR14:/x14\/a4:/g" \
		-e "s/GPR15:/x15\/a5:/g" \
		-e "s/GPR16:/x16\/a6:/g" \
		-e "s/GPR17:/x17\/a7:/g" \
		-e "s/GPR18:/x18\/s2:/g" \
		-e "s/GPR19:/x19\/s3:/g" \
		-e "s/GPR20:/x20\/s4:/g" \
		-e "s/GPR21:/x21\/s5:/g" \
		-e "s/GPR22:/x22\/s6:/g" \
		-e "s/GPR23:/x23\/s7:/g" \
		-e "s/GPR24:/x24\/s8:/g" \
		-e "s/GPR25:/x25\/s9:/g" \
		-e "s/GPR26:/x26\/s10:/g" \
		-e "s/GPR27:/x27\/s11:/g" \
		-e "s/GPR28:/x28\/t3:/g" \
		-e "s/GPR29:/x29\/t4:/g" \
		-e "s/GPR30:/x30\/t5:/g" \
		-e "s/GPR31:/x31\/t6:/g" \
		-e "s/FPR0:/f0\/ft0:/g" \
		-e "s/FPR1:/f1\/ft1:/g" \
		-e "s/FPR2:/f2\/ft2:/g" \
		-e "s/FPR3:/f3\/ft3:/g" \
		-e "s/FPR4:/f4\/ft4:/g" \
		-e "s/FPR5:/f5\/ft5:/g" \
		-e "s/FPR6:/f6\/ft6:/g" \
		-e "s/FPR7:/f7\/ft7:/g" \
		-e "s/FPR8:/f8\/fs0:/g" \
		-e "s/FPR9:/f9\/fs1:/g" \
		-e "s/FPR10:/f10\/fa0:/g" \
		-e "s/FPR11:/f11\/fa1:/g" \
		-e "s/FPR12:/f12\/fa2:/g" \
		-e "s/FPR13:/f13\/fa3:/g" \
		-e "s/FPR14:/f14\/fa4:/g" \
		-e "s/FPR15:/f15\/fa5:/g" \
		-e "s/FPR16:/f16\/fa6:/g" \
		-e "s/FPR17:/f17\/fa7:/g" \
		-e "s/FPR18:/f18\/fs2:/g" \
		-e "s/FPR19:/f19\/fs3:/g" \
		-e "s/FPR20:/f20\/fs4:/g" \
		-e "s/FPR21:/f21\/fs5:/g" \
		-e "s/FPR22:/f22\/fs6:/g" \
		-e "s/FPR23:/f23\/fs7:/g" \
		-e "s/FPR24:/f24\/fs8:/g" \
		-e "s/FPR25:/f25\/fs9:/g" \
		-e "s/FPR26:/f26\/fs10:/g" \
		-e "s/FPR27:/f27\/fs11:/g" \
		-e "s/FPR28:/f28\/ft8:/g" \
		-e "s/FPR29:/f29\/ft9:/g" \
		-e "s/FPR30:/f30\/ft10:/g" \
		-e "s/FPR31:/f31\/ft11:/g" build/prog.trace
	@$(eval list := $(shell grep -rF "PROGRAM_COUNTER:0x" build/prog.trace | sed "s/PROGRAM_COUNTER:0x//g"))
	@$(foreach item,$(list),$(call replace,$(item));)
	@echo "build/prog.trace ready for reading"

define replace
$(eval line_f :=$(shell grep -m 1 -r "PROGRAM_COUNTER:0x$(1)" build/prog.trace))
$(eval line_r :=$(shell grep -m 1 -r "$(1):" build/prog.dump))
sed "s/$(line_f)/$(line_r)/g" -i build/prog.trace
endef

build/readable:
	@make -s run TEST=$(TEST) DEBUG=1

# Define the 'test' target to compile and run a specific test
.PHONY: test
test: build
	@if [ -z ${TEST} ]; then echo -e "\033[1;31mTEST is not set\033[0m"; exit 1; fi
	@if [ ! -f tests/$(TEST) ]; then echo -e "\033[1;31mtests/$(TEST) does not exist\033[0m"; exit 1; fi
	@$(eval TEST_TYPE := $(shell echo "$(TEST)" | sed "s/.*\.//g"))
	@if [ "$(TEST_TYPE)" = "c" ]; then TEST_ARGS="lib/startup.s"; else TEST_ARGS=""; fi; \
		${RISCV64_GCC} -march=rv32imf -mabi=ilp32f -nostdlib -nostartfiles -o build/prog.elf tests/$(TEST) $$TEST_ARGS -Ilib
	@${RISCV64_OBJCOPY} -O verilog build/prog.elf build/prog.hex
	@${RISCV64_NM} build/prog.elf > build/prog.sym
	@${RISCV64_OBJDUMP} -d build/prog.elf > build/prog.dump

.PHONY: wave
wave:
# if build/prog.vcd doesn't exist, make run
	@if [ ! -f build/prog.vcd ] || [ "$(TEST)" != "$(file <build/test)" ]; then \
			make -s run TEST=$(TEST) DEBUG=1; \
	fi
	@gtkwave build/prog.vcd &

# Define the 'help' target to display usage information
.PHONY: help
help:
	@echo -e "\033[1;32mUsage:\033[0m"
	@echo -e "\033[1;35m  make help                    \033[0m# Display this help message"
	@echo -e "\033[1;35m  make clean                   \033[0m# Remove the build directory"
	@echo -e "\033[1;35m  make vivado TEST=<test>      \033[0m# Clean and run the build"
	@echo -e "\033[1;35m  make run TEST=<test>         \033[0m# Run the tests"
	@echo -e "\033[1;35m  make run TEST=<test> DEBUG=1 \033[0m# Run the tests with debug mode"
	@echo -e "\033[1;35m  make wave                    \033[0m# Open GTKWave with the VCD file"
	@echo -e ""
	@echo -e "\033[1;32mExamples:\033[0m"
	@for file in $(shell ls tests); do \
		if [ $${file##*.} = "c" ] || [ $${file##*.} = "s" ]; then \
			echo -e "\033[1;35m  make run TEST=$${file}\033[0m"; \
		fi; \
	done
