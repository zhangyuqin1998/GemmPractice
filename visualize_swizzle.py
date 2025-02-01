class Swizzle:
    def __init__(self, B, M, S):
        self.B = B
        self.M = M
        self.S = S
    
    def apply(self, offset):
        x = offset % (2 ** (self.S + self.M))
        y = offset // (2 ** (self.S + self.M))
        assert y < 2 ** self.B

        offset = x % (2 ** self.M)
        x = x // (2 ** self.M)
        x = y ^ x
        x = x * (2 ** self.M) + offset
        x = x % (2 ** (self.S + self.M))
        return x + y * (2 ** (self.S + self.M))
    
    def apply_swizzle(self, x, y, rows, cols):
        offset = x + y * cols
        new_offset = self.apply(offset)
        return (new_offset % cols, new_offset // cols)

    def apply_swizzle_v2(self, x, y, rows, cols):
        #等价V1, 但高效一点
        offset = x + y * cols
        element_per_row = 1 << (self.S + self.M)
        x = offset & (element_per_row - 1)  # 等价于 offset % (2 ** (self.S + self.M))
        y = offset >> (self.S + self.M)              # 等价于 offset // (2 ** (self.S + self.M))

        offset = x & ((1 << self.M) - 1)             # 等价于 x % (2 ** self.M)
        x = x >> self.M                              # 等价于 x // (2 ** self.M)
        x = y ^ x
        x = (x << self.M) | offset                   # 等价于 x * (2 ** self.M) + offset
        x = x & (element_per_row - 1)       # 等价于 x % (2 ** (self.S + self.M))
        new_offset = x + (y << (self.S + self.M))    # 等价于 x + y * (2 ** (self.S + self.M))
        return (new_offset % cols, new_offset // cols)

    def apply_swizzle_v3(self, x, y, rows, cols):
        #等价V1和V2, 最高效，cutlass的处理
        offset = x + y * cols

        bit_msk = (1 << self.B) - 1
        yyy_msk = bit_msk << (self.M + max(0, self.S))
        msk_sft = self.S
        if msk_sft > 0:
            new_offset = offset ^ ((offset & yyy_msk) >> msk_sft)
        else:
            new_offset =  offset ^ ((offset & yyy_msk) << -msk_sft)
        return (new_offset % cols, new_offset // cols)


def print_matrix(matrix, title):
    """
    打印矩阵
    :param matrix: 要打印的矩阵
    :param title: 矩阵标题
    """
    print(f"{title}:")
    print("=" * (4 * len(matrix[0]) + 10))
    for row in matrix:
        print(" ".join(f"({x:2},{y:2})" for x, y in row))
    print()

B = 3
M = 2
S = 3

ROWS = 16
COLS = 16

assert ROWS * COLS == 2 ** (B + M + S)

swizzle = Swizzle(B, M, S)

original_matrix = [[(x, y) for x in range(COLS)] for y in range(ROWS)]
swizzled_matrix = [[swizzle.apply_swizzle_v3(x, y, ROWS, COLS) for x in range(COLS)] for y in range(ROWS)]

print_matrix(original_matrix, "Original Matrix")
print_matrix(swizzled_matrix, f"Swizzle<B={B}, M={M}, S={S}> Matrix")