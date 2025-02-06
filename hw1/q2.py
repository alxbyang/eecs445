import numpy as np
import numpy.typing as npt


def part_a(A: npt.NDArray, B: npt.NDArray) -> npt.NDArray:
    C = np.linalg.norm(A, axis=1)**2 + (np.linalg.norm(B, axis=1)**2).reshape(-1, 1) - 2 * np.matmul(A, B.T).T    
    return np.argmin(C, axis=1)

def part_b(A: npt.NDArray, B: npt.NDArray) -> tuple[float, float]:
    C = np.dot(A, B).T / (np.linalg.norm(A, axis=1) * np.linalg.norm(B))
    return (np.argmin(C, axis=1)[0], np.min(C))


if __name__ == "__main__":
    # We've provided sample test cases for you to verify your code with. Please note that passing these test
    # cases does not guarantee that your implementation is correct. We **highly recommend** that you write
    # your own test cases apart from the basic tests provided in the cell below. Not only will this help with
    # debugging your implementation, but it helps you build strong software engineering practices.
    
    TEST_INPUT_PART_A_a = np.array([
        [0.00552212, 0.81546143, 0.70685734],
        [0.72900717, 0.77127035, 0.07404465],
        [0.35846573, 0.11586906, 0.86310343],
        [0.62329813, 0.33089802, 0.06355835],
    ])
    TEST_INPUT_PART_A_b = np.array([
        [0.31098232, 0.32518332, 0.72960618],
        [0.63755747, 0.88721274, 0.47221493],
        [0.11959425, 0.71324479, 0.76078505],
        [0.5612772,  0.77096718, 0.4937956 ],
        [0.52273283, 0.42754102, 0.02541913],
        [0.10789143, 0.03142919, 0.63641041],
    ])
    
    # TEST_INPUT_PART_A_a = np.array([
    #     [0, 1],
    #     [1, 0],
    #     [-1, 0],
    #     [0, -1]
    # ])
    # TEST_INPUT_PART_A_b = np.array([
    #     [2,1]
    #     [2,0]
    #     [-3,-1]
    #     [0,8]
    #     [2,3]
    # ])
    TEST_OUTPUT_PART_A = np.array([2, 1, 0, 1, 3, 2])
    test_a_result = np.allclose(part_a(TEST_INPUT_PART_A_a, TEST_INPUT_PART_A_b), TEST_OUTPUT_PART_A)
    print(f"Test A passed: {test_a_result}")

    TEST_INPUT_PART_B_a = np.array([
        [0.63352971, 0.53577468, 0.09028977, 0.8353025],
        [0.32078006, 0.18651851, 0.04077514, 0.59089294],
        [0.67756436, 0.01658783, 0.51209306, 0.22649578],
        [0.64517279, 0.17436643, 0.69093774, 0.38673535],
        [0.93672999, 0.13752094, 0.34106635, 0.11347352],
    ])
    TEST_INPUT_PART_B_b = np.array([
        [0.92469362],
        [0.87733935],
        [0.25794163],
        [0.65998405],
    ])
    # TEST_INPUT_PART_B_a = np.array([
    #     [1,1],
    #     [2,1],
    #     [3,1],
    #     [4,1],
    #     [1,2]
    # ])
    # TEST_INPUT_PART_B_b = np.array([
    #     [0],
    #     [1]
    # ])
    TEST_OUTPUT_PART_B = (2, 0.719627281044947)
    test_b_result = np.allclose(part_b(TEST_INPUT_PART_B_a, TEST_INPUT_PART_B_b), TEST_OUTPUT_PART_B)
    print(f"Test B passed: {test_b_result}")
