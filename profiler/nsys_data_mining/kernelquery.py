import enum

from .nsysreport import Report


""" Taken from Nvidia Night Systems 2021.1.1 /reports. """


@enum.unique
class KernelReportIndexing(enum.Enum):
    START = 0
    END = 1
    DURATION = 2
    BLKX = 7
    NAME = 18
    # -- Enum count -- #
    COUNT = 19


class CUDAKernelTrace(Report):

    usage = f"""{{SCRIPT}} -- CUDA GPU Trace

    No arguments.

    Output:
        Start : Start time of trace event in seconds
        End : Start time of trace event in seconds
        Duration : Length of event in nanoseconds
        CorrId : Correlation ID
        GrdX, GrdY, GrdZ : Grid values
        BlkX, BlkY, BlkZ : Block values
        Reg/Trd : Registers per thread
        StcSMem : Size of Static Shared Memory
        DymSMem : Size of Dynamic Shared Memory
        LocalMem : /TBE/
        ShrdMemExec: /TBE/
        Device : GPU device name and ID
        Ctx : Context ID
        Strm : Stream ID
        Name : Trace event name

    This report displays a trace of CUDA kernels and memory operations.
    Items are sorted by start time.
"""  # noqa

    query_stub = """
WITH
    {MEM_KIND_STRS_CTE}
    {MEM_OPER_STRS_CTE}
    recs AS (
        {GPU_SUB_QUERY}
    )
    SELECT
        printf('%.6f', start / 1000000000.0 ) AS "Start(sec)",
        printf('%.6f', end / 1000000000.0 ) AS "End(sec)",
        duration AS "Duration(nsec)",
        correlation AS "CorrId",
        gridX AS "GrdX",
        gridY AS "GrdY",
        gridZ AS "GrdZ",
        blockX AS "BlkX",
        blockY AS "BlkY",
        blockZ AS "BlkZ",
        regsperthread AS "Reg/Trd",
        ssmembytes AS "StcSMem",
        dsmembytes AS "DymSMem",
        localMemoryTotal AS "LocalMem",
        sharedMemoryExecuted AS "ShrdMemExec",
        device AS "Device",
        context AS "Ctx",
        stream AS "Strm",
        name AS "Name"
    FROM
            recs
    ORDER BY start;
"""

    query_kernel = """
        SELECT
            start AS "start",
            end AS "end",
            (end - start) AS "duration",
            gridX AS "gridX",
            gridY AS "gridY",
            gridZ AS "gridZ",
            blockX AS "blockX",
            blockY AS "blockY",
            blockZ AS "blockZ",
            registersPerThread AS "regsperthread",
            staticSharedMemory AS "ssmembytes",
            dynamicSharedMemory AS "dsmembytes",
            localMemoryTotal AS "localMemoryTotal",
            sharedMemoryExecuted AS "sharedMemoryExecuted",
            printf('%s (%d)', gpu.name, deviceId) AS "device",
            contextId AS "context",
            streamId AS "stream",
            dmn.value AS "name",
            correlationId AS "correlation"
        FROM
            CUPTI_ACTIVITY_KIND_KERNEL
        LEFT JOIN
            StringIds AS dmn
            ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = dmn.id
        LEFT JOIN
            TARGET_INFO_CUDA_GPU AS gpu
            USING( deviceId )
"""

    query_union = """
        UNION ALL
"""

    def setup(self):
        err = super().setup()
        if err is not None:
            return err

        sub_queries = []

        if self.table_exists("CUPTI_ACTIVITY_KIND_KERNEL"):
            sub_queries.append(self.query_kernel)

        if len(sub_queries) == 0:
            return "{DBFILE} does not contain GPU trace data."

        self.query = self.query_stub.format(
            MEM_OPER_STRS_CTE=self.MEM_OPER_STRS_CTE,
            MEM_KIND_STRS_CTE=self.MEM_KIND_STRS_CTE,
            GPU_SUB_QUERY=self.query_union.join(sub_queries),
        )


if __name__ == "__main__":
    CUDAKernelTrace.Main()
