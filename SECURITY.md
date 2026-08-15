# Security Policy

## Supported versions

The latest minor release line of each package (`milvusql`,
`milvusql-sqlalchemy`, `milvusql-django`) receives security fixes.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Instead,
use GitHub's private vulnerability reporting on this repository
("Security" tab → "Report a vulnerability"). You should receive an
initial response within 7 days.

Please include: the affected package and version, a minimal
reproduction, and the impact you believe it has.

## Scope notes

- `milvusql` renders scalar bind values into Milvus filter expressions
  itself (Milvus's server-side `{name}` templating is not usable —
  see `translate/_common.py`); string escaping is JSON-based and
  covered by tests. Filter-injection reports against that path are in
  scope and taken seriously.
- Credentials (`token`, `user:password`) are passed through to
  `pymilvus` and never logged by this library.
- Vulnerabilities in Milvus itself or `pymilvus` should be reported
  upstream to [milvus-io](https://github.com/milvus-io/milvus/security).
