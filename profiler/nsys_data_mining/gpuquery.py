#!/usr/bin/env python

""" Taken from Nvidia Night Systems 2021.1.1 /reports. """

from .nsys_sql_version import NsysSQLVersion
from .nsysreport import Report


class CUDAGPUTrace(Report):

    usage = f"""{{SCRIPT}} -- CUDA GPU Trace

    No arguments.

    Output:
        Start : Start time of trace event in seconds
        Duration : Length of event in nanoseconds
        CorrId : Correlation ID
        GrdX, GrdY, GrdZ : Grid values
        BlkX, BlkY, BlkZ : Block values
        Reg/Trd : Registers per thread
        StcSMem : Size of Static Shared Memory
        DymSMem : Size of Dynamic Shared Memory
        Bytes : Size of memory operation
        Thru : Throughput in MB per Second
        SrcMemKd : Memcpy source memory kind or memset memory kind
        DstMemKd : Memcpy destination memory kind
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
        bytes AS "Bytes",
        CASE
            WHEN bytes IS NULL
                THEN ''
            ELSE
                printf('%.3f', (bytes * 1000.0) / duration)
        END AS "Thru(MB/s)",
        srcmemkind AS "SrcMemKd",
        dstmemkind AS "DstMemKd",
        device AS "Device",
        context AS "Ctx",
        stream AS "Strm",
        name AS "Name"
    FROM
            recs
    ORDER BY start;
"""
    query_kernel_template = """
        SELECT
            start AS "start",
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
            NULL AS "bytes",
            NULL AS "srcmemkind",
            NULL AS "dstmemkind",
            NULL AS "memsetval",
            printf('%s (%d)', gpu.name, TPL_DEVICE_ID) AS "device",
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
            TPL_GPU_INFO_TABLE AS gpu
            USING( TPL_DEVICE_ID )
"""

    query_memcpy_template = """
        SELECT
            start AS "start",
            (end - start) AS "duration",
            NULL AS "gridX",
            NULL AS "gridY",
            NULL AS "gridZ",
            NULL AS "blockX",
            NULL AS "blockY",
            NULL AS "blockZ",
            NULL AS "regsperthread",
            NULL AS "ssmembytes",
            NULL AS "dsmembytes",
            bytes AS "bytes",
            msrck.name AS "srcmemkind",
            mdstk.name AS "dstmemkind",
            NULL AS "memsetval",
            printf('%s (%d)', gpu.name, TPL_DEVICE_ID) AS "device",
            contextId AS "context",
            streamId AS "stream",
            memopstr.name AS "name",
            correlationId AS "correlation"
        FROM
            CUPTI_ACTIVITY_KIND_MEMCPY AS memcpy
        LEFT JOIN
            MemcpyOperStrs AS memopstr
            ON memcpy.copyKind = memopstr.id
        LEFT JOIN
            MemKindStrs AS msrck
            ON memcpy.srcKind = msrck.id
        LEFT JOIN
            MemKindStrs AS mdstk
            ON memcpy.dstKind = mdstk.id
        LEFT JOIN
            TARGET_INFO_GPU AS gpu
            USING( TPL_DEVICE_ID )
"""

    query_memset_template = """
        SELECT
            start AS "start",
            (end - start) AS "duration",
            NULL AS "gridX",
            NULL AS "gridY",
            NULL AS "gridZ",
            NULL AS "blockX",
            NULL AS "blockY",
            NULL AS "blockZ",
            NULL AS "regsperthread",
            NULL AS "ssmembytes",
            NULL AS "dsmembytes",
            bytes AS "bytes",
            mk.name AS "srcmemkind",
            NULL AS "dstmemkind",
            value AS "memsetval",
            printf('%s (%d)', gpu.name, TPL_DEVICE_ID) AS "device",
            contextId AS "context",
            streamId AS "stream",
            '[CUDA memset]' AS "name",
            correlationId AS "correlation"
        FROM
            CUPTI_ACTIVITY_KIND_MEMSET AS memset
        LEFT JOIN
            MemKindStrs AS mk
            ON memset.memKind = mk.id
        LEFT JOIN
            TPL_GPU_INFO_TABLE AS gpu
            USING( TPL_DEVICE_ID )
"""

    query_union = """
        UNION ALL
"""

    def __init__(self, dbfile, nsys_version, args):
        if nsys_version == NsysSQLVersion.EARLY_2021:
            TPL_DEVICE_ID = "deviceId"
            TPL_GPU_INFO_TABLE = "TARGET_INFO_CUDA_GPU"
        elif nsys_version == NsysSQLVersion.MID_2021:
            TPL_DEVICE_ID = "id"
            TPL_GPU_INFO_TABLE = "TARGET_INFO_GPU"
        else:
            raise NotImplementedError(
                f"nsys SQL version {nsys_version} not implemented."
            )

        self._query_kernel = self.query_kernel_template.replace(
            "TPL_DEVICE_ID", TPL_DEVICE_ID
        ).replace("TPL_GPU_INFO_TABLE", TPL_GPU_INFO_TABLE)
        self._query_memcpy = self.query_memcpy_template.replace(
            "TPL_DEVICE_ID", TPL_DEVICE_ID
        ).replace("TPL_GPU_INFO_TABLE", TPL_GPU_INFO_TABLE)
        self._query_memset = self.query_memset_template.replace(
            "TPL_DEVICE_ID", TPL_DEVICE_ID
        ).replace("TPL_GPU_INFO_TABLE", TPL_GPU_INFO_TABLE)

        super().__init__(dbfile, nsys_version, args=args)

    def setup(self):
        err = super().setup()
        if err is not None:
            return err

        sub_queries = []

        if self.table_exists("CUPTI_ACTIVITY_KIND_KERNEL"):
            sub_queries.append(self._query_kernel)

        if self.table_exists("CUPTI_ACTIVITY_KIND_MEMCPY"):
            sub_queries.append(self._query_memcpy)

        if self.table_exists("CUPTI_ACTIVITY_KIND_MEMSET"):
            sub_queries.append(self._query_memset)

        if len(sub_queries) == 0:
            return "{DBFILE} does not contain GPU trace data."

        self.query = self.query_stub.format(
            MEM_OPER_STRS_CTE=self.MEM_OPER_STRS_CTE,
            MEM_KIND_STRS_CTE=self.MEM_KIND_STRS_CTE,
            GPU_SUB_QUERY=self.query_union.join(sub_queries),
        )


if __name__ == "__main__":
    CUDAGPUTrace.Main()
