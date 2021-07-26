import enum
import os
import sqlite3
import urllib


class NsysSQLVersion(enum.Enum):
    EARLY_2021 = 0  # >= 2021.1.0 < 2021.3.1
    MID_2021 = 1  # >= 2021.3.1


def _SQL_check_for_table(dbfile, table_name) -> bool:
    """Check for the existence of a TABLE in a given sqllite database.

    Closes the DB after check.
    """
    # Check DB file
    if not os.path.exists(dbfile):
        raise RuntimeError(f"Error_MissingDatabaseFile {dbfile}")

    # Open DB file
    dburi_query = {"mode": "ro", "nolock": "1", "immutable": "1"}
    qstr = urllib.parse.urlencode(dburi_query)
    urlstr = urllib.parse.urlunsplit(["file", "", os.path.abspath(dbfile), qstr, ""])
    try:
        dbcon = sqlite3.connect(urlstr, isolation_level=None, uri=True)
    except sqlite3.Error:
        dbcon = None
        raise RuntimeError(f"Error_InvalidDatabaseFile {dbfile}")

    # Query
    cursor = dbcon.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    )
    table_exist = False if cursor.fetchone() is None else True
    dbcon.close()
    return table_exist


def get_nsys_sql_version(dbfile) -> NsysSQLVersion:
    nsys_sql_version = NsysSQLVersion.MID_2021
    if _SQL_check_for_table(dbfile, "TARGET_INFO_CUDA_GPU"):
        nsys_sql_version = NsysSQLVersion.EARLY_2021
    return nsys_sql_version
