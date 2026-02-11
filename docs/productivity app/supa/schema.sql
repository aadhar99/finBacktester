-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- 1. PROFILES (Managed via Supabase Auth)
create table profiles (
  id uuid references auth.users not null primary key,
  email text,
  full_name text,
  avatar_url text,
  updated_at timestamptz
);

-- 2. KEY RESULTS (The "North Star" Goals)
create table key_results (
  id uuid default uuid_generate_v4() primary key,
  title text not null,
  description text,
  deadline date,
  status text default 'active' check (status in ('active', 'achieved', 'abandoned')),
  owner_id uuid references profiles(id), -- Who owns this goal?
  created_at timestamptz default now()
);

-- 3. HYPOTHESES (The Strategic Bets)
create table hypotheses (
  id uuid default uuid_generate_v4() primary key,
  kr_id uuid references key_results(id) on delete cascade,
  title text not null,
  
  -- The "Science" Metrics
  confidence_score float default 0.5 check (confidence_score >= 0.0 and confidence_score <= 1.0),
  is_validated boolean default null, -- true = working, false = failed
  
  -- The Lifecycle
  status text default 'active' check (status in ('active', 'ghost', 'graveyard', 'archived')),
  graveyard_entry_date timestamptz, -- Date it failed, for 30-day countdown
  consolidated_learnings text, -- AI Summary of why it failed/succeeded
  
  created_at timestamptz default now()
);

-- 4. EXPERIMENTS (The Kanban Tasks)
create table experiments (
  id uuid default uuid_generate_v4() primary key,
  hypothesis_id uuid references hypotheses(id) on delete cascade,
  title text not null,
  assigned_to uuid references profiles(id),
  
  -- Kanban Board Status
  status text default 'todo' check (status in ('todo', 'in_progress', 'review', 'done')),
  
  -- The Feedback Loop (Critical for AI)
  learning_note text, -- User input: "What happened?"
  learning_sentiment float, -- AI calculated: -1.0 (bad) to 1.0 (good)
  
  due_date timestamptz,
  created_at timestamptz default now()
);

-- 5. CHAT LOGS (For the AI Agent Context)
create table chat_logs (
  id uuid default uuid_generate_v4() primary key,
  user_id uuid references profiles(id),
  message text not null,
  role text default 'user' check (role in ('user', 'assistant')),
  context_data jsonb, -- Stores which KR/Hypothesis was being discussed
  created_at timestamptz default now()
);

-- Row Level Security (RLS) - Basic Setup for Household (Open to auth users)
alter table profiles enable row level security;
alter table key_results enable row level security;
alter table hypotheses enable row level security;
alter table experiments enable row level security;

create policy "Public profiles are viewable by everyone" on profiles for select using (true);
create policy "Users can insert their own profile" on profiles for insert with check (auth.uid() = id);

-- Allow all authenticated users to read/write (Household Model)
create policy "Household view KRs" on key_results for select using (auth.role() = 'authenticated');
create policy "Household edit KRs" on key_results for all using (auth.role() = 'authenticated');

-- (Repeat similar policies for hypotheses and experiments for MVP simplicity)
