""" Taken from Nvidia Night Systems 2021.1.1 /reports. """

import csv
import inspect
import os.path
import re
import sqlite3
import sys
import urllib.parse


class Report:
    class Error(Exception):
        pass

    class Error_MissingDatabaseFile(Error):
        def __init__(self, filename):
            super().__init__(f"Database file {filename} does not exist.")

    class Error_InvalidDatabaseFile(Error):
        def __init__(self, filename):
            super().__init__(
                f"Database file {filename} could not be opened and appears"
                f"to be invalid."
            )

    class Error_InvalidSQL(Error):
        def __init__(self, sql):
            super().__init__(f"Bad SQL statement: {sql}")

    EXIT_HELP = 25
    EXIT_DB = 26
    EXIT_NODATA = 27
    EXIT_SCRIPT = 28
    EXIT_INVALID_ARG = 29

    MEM_KIND_STRS_CTE = """
    MemKindStrs (id, name) AS (
    VALUES
        (0,     'Pageable'),
        (1,     'Pinned'),
        (2,     'Device'),
        (3,     'Array'),
        (4,     'Unknown')
    ),
"""

    MEM_OPER_STRS_CTE = """
    MemcpyOperStrs (id, name) AS (
    VALUES
        (0,     '[CUDA memcpy Unknown]'),
        (1,     '[CUDA memcpy HtoD]'),
        (2,     '[CUDA memcpy DtoH]'),
        (3,     '[CUDA memcpy HtoA]'),
        (4,     '[CUDA memcpy AtoH]'),
        (5,     '[CUDA memcpy AtoA]'),
        (6,     '[CUDA memcpy AtoD]'),
        (7,     '[CUDA memcpy DtoA]'),
        (8,     '[CUDA memcpy DtoD]'),
        (9,     '[CUDA memcpy HtoH]'),
        (10,    '[CUDA memcpy PtoP]'),
        (11,    '[CUDA Unified Memory memcpy HtoD]'),
        (12,    '[CUDA Unified Memory memcpy DtoH]'),
        (13,    '[CUDA Unified Memory memcpy DtoD]')
    ),
"""

    _LOAD_TABLE_QUERY = """
        SELECT name
        FROM sqlite_master
        WHERE type LIKE 'table'
           OR type LIKE 'view';
"""

    _boilerplate_statements = [
        f"pragma cache_size=-{32 * 1024}",  # Set page cache to 32MB
    ]

    short_name = None
    usage = "{SCRIPT} -- NO USAGE INFORMATION PROVIDED"
    table_checks = {}  # type: ignore
    statements = []  # type: ignore
    query = "SELECT 1 AS 'ONE'"

    def __init__(self, dbfile, nsys_version, args=[]):
        self._tables = None
        self._dbcon = None
        self._dbcur = None
        self._dbfile = dbfile
        self._args = args
        self._headers = []
        self.nsys_version = (nsys_version,)

        # Check DB file
        if not os.path.exists(self._dbfile):
            raise self.Error_MissingDatabaseFile(self._dbfile)

        # Open DB file
        dburi_query = {"mode": "ro", "nolock": "1", "immutable": "1"}
        qstr = urllib.parse.urlencode(dburi_query)
        urlstr = urllib.parse.urlunsplit(
            ["file", "", os.path.abspath(self._dbfile), qstr, ""]
        )
        try:
            self._dbcon = sqlite3.connect(urlstr, isolation_level=None, uri=True)
        except sqlite3.Error:
            self._dbcon = None
            raise self.Error_InvalidDatabaseFile(self._dbfile)

        # load tables
        try:
            cur = self._dbcon.execute(self._LOAD_TABLE_QUERY)
        except sqlite3.Error:
            raise self.Error_InvalidDatabaseFile(self._dbfile)

        self._tables = set(r[0] for r in cur.fetchall())

    def __del__(self):
        if self._dbcon is not None:
            self._dbcon.close()

    def table_exists(self, table):
        return table in self._tables

    def search_tables(self, regex_str):
        regex = re.compile(regex_str)
        matches = []
        for t in self._tables:
            if regex.search(t) is not None:
                matches.append(t)
        return matches

    def setup(self):
        for table, errmsg in self.table_checks.items():
            if not self.table_exists(table):
                return errmsg

    def get_statements(self):
        return self.statements

    def _execute_statement(self, stmt):
        if self._dbcon is None:
            raise RuntimeError(f"Called {__name__}() with invalid database connection.")

        try:
            self._dbcon.execute(stmt)
        except sqlite3.Error as err:
            return str(err)

    def run_statements(self):
        for stmt in self._boilerplate_statements:
            errmsg = self._execute_statement(stmt)
            if errmsg is not None:
                return errmsg

        for stmt in self.get_statements():
            errmsg = self._execute_statement(stmt)
            if errmsg is not None:
                return errmsg

    def get_query(self):
        return self.query

    def run_query(self):
        csvw = csv.writer(sys.stdout)
        qcur = self._dbcon.execute(self.get_query())
        qcur.arraysize = 100
        header = list(d[0] for d in qcur.description)
        csvw.writerow(header)

        rows = qcur.fetchmany()
        while rows != []:
            csvw.writerows(rows)
            rows = qcur.fetchmany()

    def start_query(self):
        if self._dbcon is None:
            raise RuntimeError(f"Called {__name__}() with invalid database connection.")
        if self._dbcur is not None:
            raise RuntimeError(f"Called {__name__}() more than once.")

        try:
            self._dbcur = self._dbcon.execute(self.get_query())
        except sqlite3.Error as err:
            return str(err)
        self._headers = list(d[0] for d in self._dbcur.description)

    def get_query_row(self):
        if self._dbcon is None:
            raise RuntimeError(f"Called {__name__}() with invalid database connection.")
        if self._dbcur is None:
            raise RuntimeError(f"Called {__name__}() without valid query.")

        row = self._dbcur.fetchone()
        if row is None:
            del self._dbcur
            self._dbcur = None
        return row

    @property
    def dbfile(self):
        return self._dbfile

    @property
    def args(self):
        return self._args

    @property
    def headers(self):
        return self._headers

    @classmethod
    def get_short_name(klass):
        if klass.short_name is None:
            klass.short_name = os.path.basename(inspect.getmodule(klass).__file__)
            if klass.short_name.endswith(".py"):
                klass.short_name = klass.short_name[0:-3]
        return klass.short_name

    @classmethod
    def get_usage(klass):
        return klass.usage.format(SCRIPT=klass.get_short_name())

    @classmethod
    def Report(klass, dbfile, nsys_version, args):
        try:
            report = klass(dbfile, nsys_version, args)
        except (
            klass.Error_MissingDatabaseFile,
            klass.Error_InvalidDatabaseFile,
        ) as err:
            return None, klass.EXIT_DB, str(err)

        errmsg = report.setup()
        if errmsg is not None:
            return None, klass.EXIT_NODATA, errmsg.format(DBFILE=report.dbfile)

        errmsg = report.run_statements()
        if errmsg is not None:
            return None, klass.EXIT_SCRIPT, errmsg

        errmsg = report.start_query()
        if errmsg is not None:
            return None, klass.EXIT_SCRIPT, errmsg

        return report, None, None

    @classmethod
    def Main(klass):
        if len(sys.argv) <= 1:
            print(klass.get_usage())
            exit(klass.EXIT_HELP)

        dbfile = sys.argv[1]
        args = sys.argv[2:]
        klass.Run(klass, dbfile, args)

    @classmethod
    def Run(klass, dbfile, nsys_version, args):
        report, exitval, errmsg = klass.Report(dbfile, nsys_version, args)
        if report is None:
            print(errmsg, file=sys.stderr)
            exit(exitval)

        # csvw = csv.writer(sys.stdout)
        # csvw.writerow(report.headers)
        results = []
        while True:
            row = report.get_query_row()
            if row is None:
                break
            # csvw.writerow(row)
            results.append(row)

        return results
