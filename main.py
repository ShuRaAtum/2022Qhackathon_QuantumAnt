import add_mult_func
from add_mult_func import *

if __name__ == "__main__":
    # Cloud(Fake Backend) : True / Simulator : False
    add_mult_func.CLOUD_SIMUL = True

    # Calculator mode : True
    DEBUG = False

    # Debug mode
    if DEBUG:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@ Debug mode Start @@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", end="\n\n")

        # print("========== 2-bit Addition (Ripple-carry) ========")
        # addition2(3,3)
        print("========== 4-bit Addition (Ripple-carry) ========")
        addition4(3, 2)
        print()

        # print("========== 4-bit Addition (Improved-Ripple-carry) ========")
        # CDKM(0b1100, 0b1010)  # b = a + b
        # print()

        print("========== 4-bit Addition (Error correction & Improved-Ripple-carry) ========")
        addition4_correction(3, 2)  # Fix 0b011, 0b0010
        print()

        # print("========== 4-bit Addition (Carry Lookahead) ========")
        # QCLA(1,2)
        # print()

        # print("========== 4-bit Multiplication (Schoolbook) ========")
        # Schoolbook_Mul(0b1100, 0b1101)
        # print()
        #
        # print("========== 4-bit Multiplication (Proposed Karatsuba) ========")
        # Karatsuba_Toffoli_Depth_1_4bit(0xd, 0xc)
        # print()
        #
        # print("========== 4-bit Multiplication (Karatsuba) ========")
        # Karatsuba_4bit(0xd, 0xc)
        # print()
        #
        # print("========== 4-bit Squaring ========")
        # Squaring(0b1110)
        # print()
        #
        # print("========== 4-bit Inversion ========")
        # Inversion(3)
        # print()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@ Debug mode Done @@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Calculator mode
    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@ Calculator mode Start @@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", end="\n\n")

        from gui import *

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@ Calculator mode Done @@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")