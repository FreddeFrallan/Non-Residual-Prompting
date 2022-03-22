import os

# GPU = input("GPU:")
GPU = "0"
# GPU = "-1"
# GPU = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

from NonResidualAttention import Inference

if __name__ == '__main__':
    Inference.main()
    print("Program Finished!")
