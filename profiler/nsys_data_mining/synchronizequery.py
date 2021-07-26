import enum

from .nsysreport import Report


""" Taken from Nvidia Night Systems 2021.1.1 /reports. """


@enum.unique
class SyncReportIndexing(enum.Enum):
    START = 0
    END = 1
    DURATION = 2
    SYNCTYPE = 3
    # -- Enum count -- #
    COUNT = 4


class SyncTrace(Report):

    usage = f"""{{SCRIPT}} -- Sync
    No arguments.

    Output:
        Start : Start time of trace event in seconds
        End : Start time of trace event in seconds
        Synctype: enum value conresponding to CUpti_ActivitySynchronizationType

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
        syncType as SyncType
    FROM
            recs
    ORDER BY start;
"""

    query_kernel = """
        SELECT
            start AS "start",
            end AS "end",
            (end - start) AS "duration",
            syncType AS "syncType"
        FROM
            CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        WHERE
            syncType==1
"""

    query_union = """
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
    SyncTrace.Main()
