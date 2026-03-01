-- DKK Code Warehouse Schema
-- Lives in DARK_FORGE.CODE_WAREHOUSE (same DB as battalion data, dedicated schema)
-- Run this once as ACCOUNTADMIN on account ymuajwd-ym41388

USE DATABASE DARK_FORGE;

CREATE SCHEMA IF NOT EXISTS CODE_WAREHOUSE;

CREATE TABLE IF NOT EXISTS CODE_WAREHOUSE.REPOSITORIES (
    id           VARCHAR(64)   PRIMARY KEY,
    name         VARCHAR(255)  NOT NULL,
    url          VARCHAR(512),
    language     VARCHAR(64),
    default_branch VARCHAR(128) DEFAULT 'main',
    last_synced  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS CODE_WAREHOUSE.REPO_FILES (
    id          VARCHAR(128)  PRIMARY KEY,   -- repo_id + ':' + path hash
    repo_id     VARCHAR(64)   NOT NULL REFERENCES CODE_WAREHOUSE.REPOSITORIES(id),
    path        VARCHAR(1024) NOT NULL,
    content     TEXT,
    language    VARCHAR(64),
    size_bytes  INTEGER,
    updated_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    embedding   VECTOR(FLOAT, 1536)          -- Snowflake Cortex embed_text() output
);

CREATE TABLE IF NOT EXISTS CODE_WAREHOUSE.COMMITS (
    id           VARCHAR(64)  PRIMARY KEY,   -- repo_id + ':' + sha
    repo_id      VARCHAR(64)  NOT NULL REFERENCES CODE_WAREHOUSE.REPOSITORIES(id),
    sha          VARCHAR(40)  NOT NULL,
    message      TEXT,
    author       VARCHAR(255),
    committed_at TIMESTAMP_NTZ
);

-- Index for fast repo lookups
CREATE INDEX IF NOT EXISTS idx_repo_files_repo ON CODE_WAREHOUSE.REPO_FILES(repo_id);
CREATE INDEX IF NOT EXISTS idx_commits_repo    ON CODE_WAREHOUSE.COMMITS(repo_id);

-- Grant read access to the warehouse role used by the Django app
GRANT USAGE ON SCHEMA CODE_WAREHOUSE TO ROLE SYSADMIN;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA CODE_WAREHOUSE TO ROLE SYSADMIN;
