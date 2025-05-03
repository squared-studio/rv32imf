.include "startup.s"

main:
    # Test 1: ebreak instruction - Breakpoint

    # The ebreak instruction is designed to cause a breakpoint exception.
    # Its precise behavior is environment-dependent (debugger, simulator, OS).
    # In a simple test, we primarily check if the instruction is recognized
    # and executes without causing immediate assembly or execution errors
    # in a basic simulation setup.

    ebreak # Execute the ebreak instruction - this should trigger a breakpoint exception.

    # In a debugging environment, execution should halt here, and control
    # should be transferred to the debugger.
    # For a basic test, we assume that if we reach the 'success' label,
    # the ebreak instruction was at least recognized and didn't lead to
    # an immediate crash in a minimal simulation.

    j success # If execution reaches here without crashing, consider it a basic success.


success:
    addi a0, zero, 0  # Success: exit code 0

exit:
    ret