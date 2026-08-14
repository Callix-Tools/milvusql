"""One-time bootstrap: create the `catalog_items` collection this
example's workflow inserts into. Run once, before starting the worker:

    python schema.py
"""

from __future__ import annotations

from config import MILVUS_TOKEN, MILVUS_URI

import milvusql

CREATE_TABLE = (
    "CREATE TABLE catalog_items ("
    "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
    "external_id VARCHAR(64), "
    "embedding VECTOR(8), "
    "category VARCHAR(64), "
    "title VARCHAR(128)"
    ") WITH (shards=1, consistency_level='Strong')"
)
CREATE_INDEX = """
CREATE INDEX idx_embedding ON catalog_items (embedding)
USING HNSW WITH (metric_type='COSINE', M=16, efConstruction=200)
"""


def main() -> None:
    conn = milvusql.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()
    try:
        cur.execute(CREATE_TABLE)
        cur.execute(CREATE_INDEX)
        cur.execute("LOAD TABLE catalog_items")
        print("catalog_items created, indexed, and loaded")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
