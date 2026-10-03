-- Memory Core schema, version 2.
--
-- Design invariants (see docs/architecture.md):
--   * This file is the only long-term source of truth.
--   * No model/vendor concept appears in any column name. The single exception
--     is memories.generated_by, which is free text, observability only, and is
--     never read by logic.
--   * Removal is logical (status / superseded_by). History is never destroyed.
--   * FTS and vector tables are DERIVED and can be rebuilt from the source
--     tables at any time.

-- Journal mode is set in code, not here: DELETE (rollback journal) is used
-- deliberately so that after a commit the entire database is one file, which is
-- what makes the byte-level model-swap assertion meaningful.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Append-only raw history. This is what makes memories regenerable.
CREATE TABLE IF NOT EXISTS events (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    session_id TEXT,
    scope      TEXT NOT NULL DEFAULT 'global',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'cli'
);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_session    ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_scope      ON events(scope);

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'global',
    subject       TEXT NOT NULL DEFAULT 'user:self',
    content       TEXT NOT NULL,
    structured    TEXT NOT NULL DEFAULT '{}',
    tags          TEXT NOT NULL DEFAULT '[]',
    source        TEXT NOT NULL DEFAULT '{}',
    confidence    REAL NOT NULL DEFAULT 1.0,
    salience      REAL NOT NULL DEFAULT 0.5,
    status        TEXT NOT NULL DEFAULT 'active',
    valid_from    TEXT,
    valid_to      TEXT,
    superseded_by TEXT REFERENCES memories(id),
    generated_by  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    revision      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_memories_status     ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_scope      ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_kind       ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

-- Explicit user declarations. Exact-key reads; a partial unique index
-- guarantees exactly one active value per (scope, key).
CREATE TABLE IF NOT EXISTS preferences (
    id            TEXT PRIMARY KEY,
    scope         TEXT NOT NULL DEFAULT 'global',
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    statement     TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 1.0,
    source        TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    superseded_by TEXT REFERENCES preferences(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_active
    ON preferences(scope, key) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS idx_preferences_key ON preferences(key);

-- "What is true now" for a project. A versioned current value, not a workflow.
CREATE TABLE IF NOT EXISTS project_state (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'global',
    state         TEXT NOT NULL DEFAULT '{}',
    note          TEXT,
    source        TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    superseded_by TEXT REFERENCES project_state(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_state_active
    ON project_state(project_id, scope) WHERE superseded_by IS NULL;

-- Derived indexes. Rebuildable from the tables above at any time; safe to drop.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    subject,
    tags,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS preferences_fts USING fts5(
    preference_id UNINDEXED,
    key,
    statement,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Derived vector index (schema v2).
--
-- NOT part of the source of truth. Every row here can be deleted and rebuilt
-- from `memories` plus an embedding provider. Keeping vectors in the same file
-- as the source tables means there is no second store to keep in sync, so the
-- "one file is the truth" guarantee still holds.
--
-- The primary key is (memory_id, model_id): an embedding is a *relationship*
-- between a memory and a model, never a column on the memory itself. Swapping
-- the embedding model therefore adds rows under a new model_id and leaves the
-- memories untouched; vectors from different models are never mixed.
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);
