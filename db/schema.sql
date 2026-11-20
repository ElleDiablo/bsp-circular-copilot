-- Part 1: Enable Vector Support
-- enables pgvector: an extension allowing PostgresSQL to store and search math AI vector embeddings
create extension if not exists vector;  

-- Part 2: Document Table (circulars)
-- Creates a table named circulars if it does not already exist
create table if not exists circulars (
    circular_no     text primary key, -- unique identifier for each document (e.g "Circular-No-1120)" , the primary key as well so no 2 documents can have the same number
    title           text not null,    -- human-readable title of the circular, cannot be left blank (not null)
    issued_on       date,             -- publication date or issue date
    source_url      text not null,    -- web link where the original document was downloaded from
    downloaded_at   timestamptz not null default now() -- Timestamp with timezone recording when script fetched file, auto fills current date/time if not specified
);

-- Part 3: The Searchable Text & Embeddings (chunks)
-- LLMs have context limits, search is more accurate when documents are broken down into small paragraphs (chunks)
create table if not exists chunks (
    id              bigserial primary key, -- auto-incrementing integer (1,2,3...) acting as unique ID for each chunk
    circular_no     text not null references circulars on delete cascade, -- foreign key linking this chunk back to its parent circular in the circulars table, if delete a circular then delete its chunks (on delete cascade)
    page            int not null,          -- page no. in PDF where this text came from, so AI can say "Cited from Page X"
    chunk_index     int not null,          -- sequence number of chunk on that document
    text            text not null,         -- actual paragraph or text content extracted from document
    embedding       vector(1024) not null, -- 1024 number representation of the text stored as an array
    unique          (circular_no, page, chunk_index) -- ensures never accidentally insert exact same chunk twice for same circular and page
);

-- Part 4: Fast Vector Search Index
create index if not exists chunks_embedding_idx         -- HNSW: hierarchical navigable small world, a fast algorithm for finding nearest vector matches in milliseconds
    on chunks using hnsw (embedding vector_cosine_ops); -- vector_cosine_ops: cosine similarity to measure relatedness of questions to chunks in database

-- Part 5: The Audit & Analytics Table (question_log)
-- records every interaction, critical for evaluating AI accuracy, monitoring latency, and tracking API costs
create table if not exists question_log (
    id              bigserial primary key,              -- auto incrementing integer IF for every logged question
    asked_at        timestamptz not null default now(), -- when the question was asked, default to current time
    question        text not null,                      -- the prompt
    answer          text not null,                      -- model's reply 
    cited           text[] not null default '{}',       -- PostgreSQL array of strings storing the list of cricular numbers cited in the answer [Circular-1120,...], Default is empty array
    latency_ms      int,                                -- how many millisecons for system to retrieve chunks and generate anser
    input_tokens    int,                                -- number of prompt and response tokens used with LLM provider
    output_tokens   int                                    -- letting you calculate the exact dollar cost of each question
);

alter table circulars    enable row level security;
alter table chunks       enable row level security;
alter table question_log enable row level security;
