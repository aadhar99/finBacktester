# Railway Deployment Guide

## Overview

Railway is the **easiest and most cost-effective** option for Phase 1-2 (paper trading).

**Cost**: ~₹1,500/month
**Setup Time**: ~10 minutes
**Complexity**: ⭐ (Very Easy)

---

## Prerequisites

1. GitHub account
2. Railway account (sign up at https://railway.app)
3. LLM API keys (at least one: Claude, Gemini, or OpenAI)

---

## Step-by-Step Setup

### 1. Push Code to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Trading system infrastructure"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/trading-system.git
git branch -M main
git push -u origin main
```

### 2. Install Railway CLI (Optional but Recommended)

```bash
# macOS
brew install railway

# Or using npm
npm install -g @railway/cli

# Login
railway login
```

### 3. Create New Railway Project

#### Option A: Using Railway Dashboard (Easiest)

1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your trading-system repository
5. Railway will auto-detect Python and start building

#### Option B: Using Railway CLI

```bash
# In your project directory
railway init

# Link to your Railway project
railway link

# Deploy
railway up
```

### 4. Add PostgreSQL Database

1. In Railway dashboard, click "New" → "Database" → "PostgreSQL"
2. Railway automatically sets `DATABASE_URL` environment variable
3. Your app can now connect to the database

### 5. Install TimescaleDB Extension

Railway's PostgreSQL doesn't have TimescaleDB by default. You have two options:

#### Option A: Use External TimescaleDB (Recommended for production)

Sign up for free tier at https://www.timescale.com/cloud

```bash
# Add external database URL as environment variable
railway variables set DATABASE_URL="postgresql://user:pass@cloud.timescale.com:5432/tsdb"
```

#### Option B: Use PostgreSQL without TimescaleDB (OK for initial testing)

The system will work fine with regular PostgreSQL for Phase 1. TimescaleDB is optional initially.

### 6. Configure Environment Variables

In Railway dashboard → Variables, add:

```bash
# LLM Providers (add at least one)
CLAUDE_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...

# Trading Configuration
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE_PCT=5.0
AUTONOMY_LEVEL=0

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Feature Flags
ENABLE_PAPER_TRADING=true
ENABLE_LIVE_TRADING=false
```

### 7. Initialize Database Schema

After first deployment, run migrations:

```bash
# Using Railway CLI
railway run python -c "
import asyncio
from utils.database import DatabaseManager
import os

async def init():
    db = DatabaseManager(os.getenv('DATABASE_URL'))
    await db.connect()
    await db.initialize_schema()
    print('✅ Schema initialized')
    await db.disconnect()

asyncio.run(init())
"
```

### 8. Deploy!

Railway automatically deploys on every git push to main branch.

```bash
git add .
git commit -m "Configure for Railway"
git push origin main
```

Railway will:
- Build your Docker image
- Run tests
- Deploy to production
- Provide a public URL (if you have a web service)

---

## Monitoring

### View Logs

```bash
# Using CLI
railway logs

# Or in dashboard
# Go to your service → Logs tab
```

### Check Database

```bash
# Connect to Railway PostgreSQL
railway run psql $DATABASE_URL

# Run queries
SELECT * FROM portfolio_state;
SELECT * FROM active_trades;
```

### View Metrics

Railway dashboard shows:
- CPU usage
- Memory usage
- Network traffic
- Build/deployment history

---

## Cost Optimization

### Free Trial
- Railway offers $5/month free credit
- Great for initial testing

### Paid Plan (~₹1,500/month)
```
PostgreSQL (1 GB RAM):     ₹800/month
Web Service (512 MB RAM):  ₹600/month
Egress (10 GB):            ₹100/month
───────────────────────────────────
TOTAL:                     ₹1,500/month
```

### Cost Saving Tips
1. **Use Gemini Free Tier** for LLM (0 cost up to 1M tokens/day)
2. **Start small**: 512 MB RAM is enough for paper trading
3. **Monitor usage**: Railway shows real-time cost estimates
4. **Scale only when needed**: Upgrade to 1 GB RAM only if necessary

---

## Troubleshooting

### Build Fails

**Error**: `Could not find a version that satisfies requirement ta-lib`

**Solution**: ta-lib requires system libraries. Add to `nixpacks.toml`:

```toml
# Create nixpacks.toml in project root
[phases.setup]
nixPkgs = ["python311", "ta-lib"]
```

### Database Connection Fails

**Error**: `could not connect to server`

**Solution**: Check DATABASE_URL format:
```bash
railway variables get DATABASE_URL

# Should be:
postgresql://postgres:password@containers-us-west-xxx.railway.app:5432/railway
```

### App Crashes on Start

**Solution**: Check logs
```bash
railway logs --tail 100
```

Common issues:
- Missing environment variables
- Database schema not initialized
- LLM API key invalid

---

## Upgrading to GCP Later

When you're ready for production (Phase 3-4), migrate to GCP:

1. **Export data**:
   ```bash
   railway run pg_dump $DATABASE_URL > backup.sql
   ```

2. **Import to GCP**:
   ```bash
   psql -h GCP_DB_HOST -U user -d dbname < backup.sql
   ```

3. **Update environment variables** to point to GCP
4. **Redeploy**

Railway is perfect for getting started quickly. Migrate to GCP when you need:
- Higher reliability (99.95% SLA)
- India region (lower latency)
- Advanced features (Cloud Run, BigQuery, etc.)

---

## Next Steps

After successful Railway deployment:

1. ✅ Verify database connection
2. ✅ Run paper trading for 1 week
3. ✅ Monitor costs and performance
4. ✅ Test LLM provider fallbacks
5. ✅ Set up Telegram alerts (optional)
6. ✅ Implement Phase 1 agents

**Railway Dashboard**: https://railway.app/dashboard
**Railway Docs**: https://docs.railway.app/
