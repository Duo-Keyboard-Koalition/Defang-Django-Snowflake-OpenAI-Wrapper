-- DKK Code Warehouse Schema
-- Extends existing DKK.COMMUNITY tables rather than creating a new database.
-- GITHUB_REPOS already has 53 rows — we add columns and create sibling tables.
-- Run as ACCOUNTADMIN on account ymuajwd-ym41388

USE DATABASE DKK;
USE SCHEMA COMMUNITY;

-- ── Step 1: Extend GITHUB_REPOS with sync tracking ──────────────────────────
-- (safe: these columns may not exist yet)
ALTER TABLE COMMUNITY.GITHUB_REPOS ADD COLUMN IF NOT EXISTS DEFAULT_BRANCH VARCHAR(128) DEFAULT 'main';
ALTER TABLE COMMUNITY.GITHUB_REPOS ADD COLUMN IF NOT EXISTS LAST_INDEXED   TIMESTAMP_NTZ;
ALTER TABLE COMMUNITY.GITHUB_REPOS ADD COLUMN IF NOT EXISTS URL            VARCHAR(512);

-- ── Step 2: Code file store with embeddings ──────────────────────────────────
CREATE TABLE IF NOT EXISTS COMMUNITY.REPO_FILES (
    id          VARCHAR(256)  PRIMARY KEY,   -- REPO_ID||':'||sha256(path)
    repo_id     NUMBER(38,0)  NOT NULL,      -- FK to GITHUB_REPOS.REPO_ID
    path        VARCHAR(2048) NOT NULL,
    content     TEXT,
    language    VARCHAR(64),
    size_bytes  INTEGER,
    sha         VARCHAR(40),                 -- blob SHA for change detection
    updated_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    embedding   VECTOR(FLOAT, 1536)          -- from SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', content)
);

-- ── Step 3: Commit history ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS COMMUNITY.REPO_COMMITS (
    id           VARCHAR(128)  PRIMARY KEY,  -- repo_id||':'||sha
    repo_id      NUMBER(38,0)  NOT NULL,
    sha          VARCHAR(40)   NOT NULL,
    message      TEXT,
    author       VARCHAR(255),
    committed_at TIMESTAMP_NTZ
);

-- ── Step 4: Indexes ──────────────────────────────────────────────────────────
-- Snowflake doesn't use traditional indexes for query optimization,
-- but clustering keys help for large tables.
ALTER TABLE COMMUNITY.REPO_FILES    CLUSTER BY (repo_id);
ALTER TABLE COMMUNITY.REPO_COMMITS  CLUSTER BY (repo_id);

-- ── Step 5: Grants ───────────────────────────────────────────────────────────
GRANT SELECT, INSERT, UPDATE ON TABLE COMMUNITY.REPO_FILES   TO ROLE SYSADMIN;
GRANT SELECT, INSERT, UPDATE ON TABLE COMMUNITY.REPO_COMMITS TO ROLE SYSADMIN;

-- ── Usage notes ──────────────────────────────────────────────────────────────
-- Vector search (cursor_app RAG):
--   SELECT path, content,
--          VECTOR_COSINE_SIMILARITY(embedding, SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', :query)) AS score
--   FROM COMMUNITY.REPO_FILES
--   WHERE repo_id = :repo_id
--   ORDER BY score DESC LIMIT 10;
--
-- Embed a file on ingest:
--   UPDATE COMMUNITY.REPO_FILES
--   SET embedding = SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', content)
--   WHERE id = :id;
