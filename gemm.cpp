// gemm.cpp
//
// Simple N x N GEMM kernel for Vitis HLS / Vitis.
//
// C = A * B
//
// Loops are labeled Row / Col / K so Tcl can apply pipeline / unroll
// directives to them.

#include <ap_int.h>
#include <hls_stream.h>

#define N 64       // You can change this; keep it moderate at first.
typedef float data_t;

void gemm(
    const data_t A[N][N],
    const data_t B[N][N],
    data_t       C[N][N]
) {
#pragma HLS INTERFACE m_axi     port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=C offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Naive triple-nested loops.
    // We will tweak performance via HLS directives from Tcl.

Row:
    for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE off  // <-- disable auto pipeline on Row
    Col:
        for (int j = 0; j < N; ++j) {
        #pragma HLS PIPELINE off  // <-- disable auto pipeline on Row
            data_t sum = 0;
        K:
            for (int k = 0; k < N; ++k) {
            #pragma HLS PIPELINE off  // <-- disable auto pipeline on Row
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
