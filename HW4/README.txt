DENSE:
To compile:
nvcc dense_matmul_opt.cu -o dense_matmul_opt
To run:
./dense_matmul_opt

SPARSE:
To compile:
nvcc sparse_matmul_opt.cu -lcusparse -o sparse_matmul_opt
Example cl runs:
./sparse_matmul_opt 494_bus.mtx 494_bus.mtx
./sparse_matmul_opt 1138_bus.mtx 1138_bus.mtx
./sparse_matmul_opt 685_bus.mtx 685_bus.mtx
