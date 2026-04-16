import numpy as np


def main():
    # Create a simple 2D array
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print("Hello from uv!")
    print(f"Here is a NumPy matrix:\n{data}")
    print(f"The sum is: {np.sum(data, axis=1)}")


if __name__ == "__main__":
    main()
