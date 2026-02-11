#!/bin/bash

# 1. Initialize Next.js 14
# Using --yes to use defaults, --no-src-dir to keep structure flat as per user pref (implied by file paths like supa/schema.sql)
npx create-next-app@latest . --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm --yes

# 2. Install Dependencies
# Supabase
npm install @supabase/supabase-js @supabase/ssr

# Styling & Icons
npm install lucide-react clsx tailwind-merge next-themes

# UI/UX Libraries
npm install @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities
npm install reactflow

# AI
npm install ai @ai-sdk/anthropic zod

# 3. Initialize Shadcn UI
# We need to initialize shadcn-ui after next.js is set up
# Running with -y might use defaults.
npx shadcn-ui@latest init --yes

echo "Project setup and dependencies installed successfully!"
