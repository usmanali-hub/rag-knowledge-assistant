# SQL indexes

An index is a separate structure, usually a B-tree, that lets the database find rows without scanning the whole table. Queries that filter or join on an indexed column can drop from seconds to milliseconds on large tables. Indexes cost disk space and slow down inserts and updates, because each write must update them too. A composite index on two columns helps queries that filter on the leftmost column first. Use the query planner output, shown by EXPLAIN, to confirm that an index is actually used before adding more.
