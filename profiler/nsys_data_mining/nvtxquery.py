#!/usr/bin/env python

""" Adapted from Nvidia Night Systems 2021.1.1 /reports. """

import enum

from .nsysreport import Report


@enum.unique
class NVTXReportIndexing(enum.Enum):
    START = 0
    END = 1
    DURATION = 2
    TEXTID = 3
    TEXT = 4


class CUDANVTXTrace(Report):

    usage = f"""{{SCRIPT}} -- CUDA GPU Trace

    No arguments.

    Output:
        Start : Start time of trace event in seconds
        End: End time of trace event in seconds
        Duration: duration in nanoseconds
        TextId: NVTX automatic text (for MPI)
        Text : user tagged text

    /TBE/
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
        textid AS "TextId(auto)",
        text AS "Text(user)"
    FROM
            recs
    ORDER BY start;
"""

    nvtx_kernel = """
        SELECT
            start AS "start",
            end AS "end",
            (end - start) AS "duration",
            tid.value AS "textid",
            text AS "text"
        FROM
            NVTX_EVENTS
        LEFT JOIN
            StringIds AS tid
            ON NVTX_EVENTS.textId = tid.id
"""

    query_union = """
        UNION ALL
"""

    def setup(self):
        err = super().setup()
        if err is not None:
            return err

        sub_queries = []

        if self.table_exists("NVTX_EVENTS"):
            sub_queries.append(self.nvtx_kernel)

        if len(sub_queries) == 0:
            return "{DBFILE} does not contain GPU trace data."

        self.query = self.query_stub.format(
            MEM_OPER_STRS_CTE=self.MEM_OPER_STRS_CTE,
            MEM_KIND_STRS_CTE=self.MEM_KIND_STRS_CTE,
            GPU_SUB_QUERY=self.query_union.join(sub_queries),
        )


if __name__ == "__main__":
    CUDANVTXTrace.Main()
