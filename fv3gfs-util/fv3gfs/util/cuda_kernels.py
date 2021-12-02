# flake8: noqa

try:
    import cupy as cp
except ImportError:
    cp = None


pack_scalar_f64_kernel = (
    None
    if cp is None
    else cp.RawKernel(
        r"""
        extern "C" __global__
        void pack_scalar_f64(const double* i_sourceArray,
                            const int* i_indexes,
                            const int i_nIndex,
                            const int i_offset,
                            double* o_destinationBuffer)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid>=i_nIndex)
            {
                return;
            }

            o_destinationBuffer[i_offset+tid] = i_sourceArray[i_indexes[tid]];
        }

        """,
        "pack_scalar_f64",
    )
)
"""Pack into o_destinationBuffer data from i_sourceArray.

    The indexation into i_sourceArray is stored in i_indexes.
    i_offset is the offset in the destination buffer.
    i_nIndex allows to protect from out-of-bound read in kernel.

    tid is the global unique index calculated from the CUDA scheduler inner data.
"""


unpack_scalar_f64_kernel = (
    None
    if cp is None
    else cp.RawKernel(
        r"""
            extern "C" __global__
            void unpack_scalar_f64(const double* i_sourceBuffer,
                                const int* i_indexes,
                                const int i_nIndex,
                                const int i_offset,
                                double* o_destinationArray)
            {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid>=i_nIndex)
                    return;

                o_destinationArray[i_indexes[tid]] = i_sourceBuffer[i_offset+tid];
            }

            """,
        "unpack_scalar_f64",
    )
)
"""Unpack into o_destinationArray data from i_sourceBuffer.

The indexation into o_destinationArray is stored in i_indexes.
i_offset is the offset in the source buffer.
i_nIndex allows to protect from out-of-bound read in kernel.

tid is the global unique index calculated from the CUDA scheduler inner data.
"""


# Expect rotate >= 0 in [0:4[
pack_vector_f64_kernel = (
    None
    if cp is None
    else cp.RawKernel(
        r"""
    extern "C" __global__
    void pack_vector_f64(const double* i_sourceArrayX,
                        const double* i_sourceArrayY,
                        const int* i_indexesX,
                        const int* i_indexesY,
                        const int i_nIndexX,
                        const int i_nIndexY,
                        const int i_offset,
                        const int i_rotate,
                        double* o_destinationBuffer)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid>=i_nIndexX+i_nIndexY)
            return;

        if (i_rotate == 0)
        {
            //pass
            if (tid<i_nIndexX)
                o_destinationBuffer[i_offset+tid] = i_sourceArrayX[i_indexesX[tid]];
            else
                o_destinationBuffer[i_offset+tid] = i_sourceArrayY[i_indexesY[tid-i_nIndexX]];
        }
        else if (i_rotate == 1)
        {
            //data[0], data[1] = data[1], -data[0]
            if (tid<i_nIndexY)
                o_destinationBuffer[i_offset+tid] = i_sourceArrayY[i_indexesY[tid]];
            else
                o_destinationBuffer[i_offset+tid] = -1.0 * i_sourceArrayX[i_indexesX[tid-i_nIndexY]];
        }
        else if (i_rotate == 2)
        {
            //data[0], data[1] = -data[0], -data[1]
            if (tid<i_nIndexX)
                o_destinationBuffer[i_offset+tid] = -1.0 * i_sourceArrayX[i_indexesX[tid]];
            else
                o_destinationBuffer[i_offset+tid] = -1.0 * i_sourceArrayY[i_indexesY[tid-i_nIndexX]];
        }
        else if (i_rotate == 3)
        {
            //data[0], data[1] = -data[1], data[0]
            if (tid<i_nIndexY)
                o_destinationBuffer[i_offset+tid] = -1.0 * i_sourceArrayY[i_indexesY[tid]];
            else
                o_destinationBuffer[i_offset+tid] = i_sourceArrayX[i_indexesX[tid-i_nIndexY]];
        }

    }

    """,
        "pack_vector_f64",
    )
)
"""Pack into o_destinationBuffer data from i_sourceArrayX/Y.

    The indexation into i_sourceArrayX/Y is stored in i_indexesX/Y.
    i_offset is the offset in the destination buffer.
    i_nIndexX/Y allows to protect from out-of-bound read in kernel.
    i_rotate refers to the rotation that needs to be applied prior to assignment.

    tid is the global unique index calculated from the CUDA scheduler inner data.
"""


unpack_vector_f64_kernel = (
    None
    if cp is None
    else cp.RawKernel(
        r"""
        extern "C" __global__
        void unpack_vector_f64(const double* i_sourceBuffer,
                            const int* i_indexesX,
                            const int* i_indexesY,
                            const int i_nIndexX,
                            const int i_nIndexY,
                            const int i_offset,
                            double* o_destinationArrayX,
                            double* o_destinationArrayY)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid<i_nIndexX)
                o_destinationArrayX[i_indexesX[tid]] = i_sourceBuffer[i_offset+tid];
            else if (tid<i_nIndexX+i_nIndexY)
                o_destinationArrayY[i_indexesY[tid-i_nIndexX]] = i_sourceBuffer[i_offset+tid];
        }

        """,
        "unpack_vector_f64",
    )
)
"""Unpack into o_destinationArrayX/Y data from i_sourceBuffer.

    The indexation into o_destinationArrayX/Y is stored in i_indexesX/Y.
    i_offset is the offset in the source buffer.
    i_nIndexX/Y allows to protect from out-of-bound read in kernel.

    tid is the global unique index calculated from the CUDA scheduler inner data.
"""
