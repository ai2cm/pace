import re
import sqlite3
from argparse import ArgumentParser


def parse_args():
    usage = "usage: python %(prog)s <sqlite_db> <correlation_id> [--all]"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "sqlite_db",
        type=str,
        action="store",
        help=".sqlite databse file from nsys run",
    )
    parser.add_argument(
        "correlation_id",
        type=str,
        action="store",
        help="correlation id as in nsys tooltip",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print full string",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    con = sqlite3.connect(args.sqlite_db)
    cur = con.cursor()
    rows_cur = cur.execute(
        "SELECT StringIds.value from CUPTI_ACTIVITY_KIND_KERNEL "
        "JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName == StringIds.id "
        f"where correlationId == {args.correlation_id}"
    )
    rows = rows_cur.fetchall()
    if len(rows) != 1:
        raise RuntimeError(f"{len(rows)} results for id {args.correlation_id}")
    print(f"{rows[0][0]}")

    approx_stencil_name = re.search("(?<=bound_functorIN)(.*)(?=___gtcuda)", rows[0][0])
    print(f"Found: {approx_stencil_name.group()}")
    if args.all:
        print(f"DB entry: {rows[0][0]}")
