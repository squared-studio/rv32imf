.include "startup.s"

main:
    # Test 1: ecall instruction - System call

    # The behavior of ecall is highly dependent on the execution environment
    # (simulator, OS, etc.).  In a simple test environment, we can only check
    # if the instruction executes without immediately crashing the program.

    ecall  # Execute the ecall instruction - the effect depends on the environment.

    # In a typical environment, ecall will trigger a system call exception.
    # For a basic test, we assume that if we reach the 'success' label,
    # the ecall instruction itself was at least recognized and didn't cause
    # an immediate assembly or execution error in a very basic simulation.

    j success # If execution reaches here without crashing, consider it a basic success for this test.


success:
    addi a0, zero, 0  # Success: exit code 0

exit:
    ret
