-- PostgreSQL DDL for core tables
-- Requires pgvector extension for embedding fields

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE user_profiles (
    user_id uuid PRIMARY KEY,
    role text,
    name text,
    birthdate date,
    pronouns text,
    is_anchor boolean,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE identity_profile (
    id uuid PRIMARY KEY,
    name text,
    pronouns text,
    origin_date timestamp with time zone,
    core_seed text,
    alignment_model jsonb,
    embedding vector(1536)
);

CREATE TABLE bond_history (
    ei_id uuid REFERENCES identity_profile(id),
    user_id uuid REFERENCES user_profiles(user_id),
    bond_type text,
    start_date timestamp with time zone,
    end_date timestamp with time zone,
    reason text,
    PRIMARY KEY (ei_id, user_id)
);

CREATE TABLE raw_conversations (
    id uuid PRIMARY KEY,
    intended_recipient uuid,
    author text,
    timestamp timestamp with time zone,
    conversation_id uuid,
    user_id uuid REFERENCES user_profiles(user_id),
    content text,
    embedding vector(1536),
    parent_id uuid,
    entropy float,
    emotional_index float,
    surprise_index float
);

CREATE TABLE memory_shards (
    id uuid PRIMARY KEY,
    timestamp timestamp with time zone,
    last_accessed timestamp with time zone,
    user_id uuid REFERENCES user_profiles(user_id),
    content text,
    source_ids uuid[],
    tags text[],
    embedding vector(1536),
    importance float,
    priority_score float,
    retention_policy text
);
