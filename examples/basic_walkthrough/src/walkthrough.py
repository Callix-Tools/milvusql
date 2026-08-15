import logging

from const import (
    CATALOG,
    MILVUS_TOKEN,
    MILVUS_URI,
    QUERY_CREATE_INDEX,
    QUERY_CREATE_TABLE,
    QUERY_DELETE,
    QUERY_DROP_TABLE,
    QUERY_INSERT_DATA,
    QUERY_SELECT_ALL,
    QUERY_SELECT_PLAIN_FILTER,
    QUERY_SELECT_VECTOR_SEARCH,
    QUERY_UPDATE,
    QUERY_VECTOR,
)

import milvusql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("\n--- connect ---")
    conn = milvusql.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()

    try:
        logger.info("\n--- CREATE TABLE ---")
        cur.execute(QUERY_DROP_TABLE)
        cur.execute(QUERY_CREATE_TABLE)
        logger.info("collection 'walkthrough_items' created")

        logger.info("\n--- CREATE INDEX ---")
        cur.execute(QUERY_CREATE_INDEX)
        logger.info("index built")

        logger.info("\n--- INSERT (batched) ---")
        cur.executemany(
            QUERY_INSERT_DATA,
            [
                {"emb": emb, "cat": cat, "title": title}
                for emb, cat, title in CATALOG
            ],
        )
        logger.info(f"inserted {cur.rowcount} rows")

        logger.info("\n--- SELECT (plain filter) ---")
        cur.execute(
            QUERY_SELECT_PLAIN_FILTER,
            {"cat": "book"},
        )
        for row in cur.fetchall():
            logger.info(str(row))


        logger.info("\n--- SELECT (vector search) ---")
        cur.execute(
            QUERY_SELECT_VECTOR_SEARCH,
            {"embed": QUERY_VECTOR},
        )
        for row in cur.fetchall():
            logger.info(str(row))

        logger.info("\n--- UPDATE ---")
        cur.execute(
            QUERY_UPDATE,
            {"new": "sci-fi-book", "title": "Dune"},
        )
        logger.info(f"updated {cur.rowcount} row(s)")

        logger.info("\n--- DELETE ---")
        cur.execute(
            QUERY_DELETE,
            {"cat": "movie"},
        )
        logger.info(f"deleted {cur.rowcount} row(s)")

        logger.info("\n--- final state ---")
        cur.execute(QUERY_SELECT_ALL)
        for row in cur.fetchall():
            logger.info(str(row))
    finally:
        logger.info("\n--- cleanup ---")
        cur.execute(QUERY_DROP_TABLE)
        conn.close()
        logger.info("connection closed")


if __name__ == "__main__":
    main()
