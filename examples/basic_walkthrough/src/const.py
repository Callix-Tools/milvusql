MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"  # noqa: S105

CATALOG = [
    # (embedding, category, title)
    (
        [0.10, 0.12, 0.11, 0.09, 0.10, 0.13, 0.11, 0.10],
        "book", "Dune"
    ),
    (
        [0.11, 0.10, 0.12, 0.10, 0.09, 0.11, 0.10, 0.12],
        "book", "neuromancer"
    ),
    (
        [0.91, 0.88, 0.90, 0.92, 0.89, 0.91, 0.90, 0.88],
        "movie", "Blade Runner"
    ),
    (
        [0.90, 0.92, 0.89, 0.91, 0.90, 0.88, 0.91, 0.90],
        "movie", "The Matrix"
    ),
]

QUERY_DROP_TABLE = "DROP TABLE IF EXISTS walkthrough_items"
QUERY_CREATE_TABLE = """
CREATE TABLE walkthrough_items (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    embedding VECTOR(8),
    category VARCHAR(64),
    title VARCHAR(128)
) WITH (shards=1, consistency_level='Strong')
"""

QUERY_CREATE_INDEX = """
CREATE INDEX idx_embedding ON walkthrough_items (embedding)
USING HNSW WITH (metric_type='COSINE', M=16, efConstruction=200)
"""

QUERY_INSERT_DATA = """
INSERT INTO walkthrough_items (embedding, category, title)
VALUES (:emb, :cat, :title)
"""

QUERY_SELECT_PLAIN_FILTER = """
SELECT id, title FROM walkthrough_items
WHERE category = :cat LIMIT 10
"""

QUERY_SELECT_VECTOR_SEARCH = """
SELECT id, title, category FROM walkthrough_items
ORDER BY embedding <=> :embed LIMIT 3
"""

QUERY_UPDATE = """
UPDATE walkthrough_items SET category = :new
WHERE title = :title
"""

QUERY_DELETE = """
DELETE FROM walkthrough_items WHERE category = :cat
"""

QUERY_SELECT_ALL = """
SELECT id, title, category FROM walkthrough_items
"""

QUERY_VECTOR = [0.10, 0.11, 0.10, 0.10, 0.11, 0.12, 0.10, 0.11]
