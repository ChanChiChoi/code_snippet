// SOME PRECAUTIONS:
// 如果是ROW-MAJOR 矩阵相乘 C = A * B,
// 只需要逆序调用CUBLAS的api: cublasSegemm(B, A)!

// CUBLAS 是使用column-major存储的，但是C/C++ 使用row-major存储.
// 当矩阵指针传递给CUBLAS, 内存布局就会从row-major 到 column-major,
// 这相当于一个潜在的转置(transpose).

// 所以对于 row-major的 C/C++矩阵 A, B,就算是最简单的矩阵相乘C = A * B, 
// 也不能直接用输入时候的顺序 cublasSgemm(A, B)，因为存在潜在转置，
// cublasSegemm(A, B) 的结果其实是 A(T) * B(T).
// 如果col(A(T)) != row(B(T)), 也就是 row(A) != col(B),那么 A(T) 和 B(T) 就无法相乘
// 而且如果 A(T) 和 B(T)是可相乘的,那么结果 C 是一个column-based 的cublas矩阵
// 这意味着要想得到C/C++中的 C(T)，就需要额外的转置将结果转成row-based的 C/C++矩阵

// 为了解决这个问题，我们是为了得到 C, 一个row-major的矩阵
// 在cublas格式中，其实是C(T) (因为有个潜在转置).
// C = A * B, 所以 C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// 所以在输入时候，我们不需要额外的转置代码，只需要调整输入位置就行
//
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

