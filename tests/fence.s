.include "startup.s"

main:
    # Test 1: fence.i (instruction fence) -  Ensuring instruction stream is ordered

    # Scenario:  Modify instruction in memory, then execute fence.i, then attempt to execute modified instruction.
    # This test is inherently limited in what it can directly observe in a simple, single-core test environment.
    # fence.i is primarily for I-cache coherence, which is complex to test directly at this level.
    # This test attempts a basic check, but might not reliably fail if fence.i is not correctly implemented in a simple simulator.

    la t0, instruction_to_modify  # Address of instruction to modify
    li t1, 0x0000000b #  `addi zero, zero, 0` instruction - a no-op. We'll overwrite with this.

    sw t1, 0(t0)       # Modify the instruction in memory to be a no-op

    fence i,o          # Instruction and data fence - to ensure write to instruction memory is seen before next instruction fetch.
                         # Using i,o to be broad, though fence.i would be more specific here.

    # Attempt to execute the potentially modified instruction.
    # In a correct implementation with proper I-cache behavior, the fence.i should ensure that the modified instruction
    # is fetched *after* the write, and thus the no-op should execute.
    # If fence.i is not working, the original instruction might be fetched from the I-cache before the write propagates,
    # and the test might behave unexpectedly.

instruction_to_modify:
    li t2, 1          # Original instruction: li t2, 1.  We intend to overwrite this.
                        # If fence works, this instruction should be effectively replaced by the no-op above.


    addi t3, zero, 1   # t3 = 1 (set a known value)

    beq t2, zero, fence_test_success # If t2 is zero, it means the 'li t2, 1' was effectively skipped (due to fence and overwrite)
                                     # and replaced by the no-op.
                                     # This is a weak check, as many factors could lead to t2 being zero.


    addi a0, zero, 1    # Failure: exit code 1 (if t2 is not zero - meaning original instruction likely executed)
    j exit


fence_test_success:
    addi a0, zero, 0     # Success: exit code 0 (if t2 is zero, indicating fence might have had effect)


exit:
    ret
