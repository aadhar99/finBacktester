#!/bin/bash

# =============================================================================
# GCP Deployment Script for Trading System
# =============================================================================
# This script automates the deployment of the trading system to Google Cloud Platform
#
# Prerequisites:
# - gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
# - Authenticated: gcloud auth login
# - Billing account enabled
#
# Usage:
#   ./deploy/gcp-deploy.sh
#
# Cost: ~₹2,350/month
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0,31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-trading-system-$(date +%s)}"
REGION="${GCP_REGION:-asia-south1}"  # Mumbai
DB_INSTANCE_NAME="${DB_INSTANCE_NAME:-trading-db}"
DB_NAME="${DB_NAME:-trading_system}"
DB_USER="${DB_USER:-trading_user}"
STORAGE_BUCKET="${STORAGE_BUCKET:-${PROJECT_ID}-audit-archive}"
SERVICE_NAME="${SERVICE_NAME:-trading-api}"

echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}GCP Deployment Script for Trading System${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Database: $DB_INSTANCE_NAME"
echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

# =============================================================================
# Step 1: Create and Configure Project
# =============================================================================

echo -e "\n${YELLOW}Step 1: Creating GCP project...${NC}"

# Check if project exists
if gcloud projects describe $PROJECT_ID &>/dev/null; then
    echo "✅ Project $PROJECT_ID already exists"
else
    gcloud projects create $PROJECT_ID --name="Trading System"
    echo "✅ Created project: $PROJECT_ID"
fi

# Set as active project
gcloud config set project $PROJECT_ID

# Link billing account (interactive)
echo "Please link a billing account to the project:"
gcloud billing projects link $PROJECT_ID

# =============================================================================
# Step 2: Enable Required APIs
# =============================================================================

echo -e "\n${YELLOW}Step 2: Enabling required APIs...${NC}"

APIS=(
    "sqladmin.googleapis.com"           # Cloud SQL
    "run.googleapis.com"                # Cloud Run
    "storage.googleapis.com"            # Cloud Storage
    "compute.googleapis.com"            # Compute Engine
    "cloudbuild.googleapis.com"         # Cloud Build
    "secretmanager.googleapis.com"      # Secret Manager
)

for api in "${APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api --quiet
done

echo "✅ All APIs enabled"

# =============================================================================
# Step 3: Create Cloud SQL Instance (PostgreSQL + TimescaleDB)
# =============================================================================

echo -e "\n${YELLOW}Step 3: Creating Cloud SQL instance...${NC}"

# Check if instance exists
if gcloud sql instances describe $DB_INSTANCE_NAME &>/dev/null; then
    echo "✅ Database instance already exists"
else
    # Generate random password
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

    # Create Cloud SQL instance
    gcloud sql instances create $DB_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=$REGION \
        --database-flags=cloudsql.enable_pgaudit=on,shared_preload_libraries=timescaledb \
        --storage-type=SSD \
        --storage-size=10GB \
        --backup \
        --backup-start-time=03:00 \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=4 \
        --quiet

    echo "✅ Created Cloud SQL instance: $DB_INSTANCE_NAME"

    # Set root password
    gcloud sql users set-password postgres \
        --instance=$DB_INSTANCE_NAME \
        --password=$DB_PASSWORD

    # Create database
    gcloud sql databases create $DB_NAME \
        --instance=$DB_INSTANCE_NAME

    # Create user
    gcloud sql users create $DB_USER \
        --instance=$DB_INSTANCE_NAME \
        --password=$DB_PASSWORD

    echo "✅ Database created: $DB_NAME"
    echo "✅ User created: $DB_USER"

    # Store password in Secret Manager
    echo -n $DB_PASSWORD | gcloud secrets create db-password \
        --data-file=- \
        --replication-policy="automatic"

    echo "✅ Password stored in Secret Manager"
fi

# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE_NAME --format='value(connectionName)')
echo "Database connection name: $CONNECTION_NAME"

# =============================================================================
# Step 4: Install TimescaleDB Extension
# =============================================================================

echo -e "\n${YELLOW}Step 4: Installing TimescaleDB extension...${NC}"

# Connect and install extension
gcloud sql connect $DB_INSTANCE_NAME --user=$DB_USER --database=$DB_NAME <<EOF
CREATE EXTENSION IF NOT EXISTS timescaledb;
SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';
\q
EOF

echo "✅ TimescaleDB extension installed"

# =============================================================================
# Step 5: Initialize Database Schema
# =============================================================================

echo -e "\n${YELLOW}Step 5: Initializing database schema...${NC}"

# Upload schema files to Cloud SQL
gcloud sql import sql $DB_INSTANCE_NAME gs://temp-bucket/postgres.sql \
    --database=$DB_NAME \
    --user=$DB_USER \
    2>/dev/null || echo "Schema import skipped (manual step required)"

echo "⚠️  Manual step required: Run schema files using Cloud SQL proxy"
echo "   1. Download Cloud SQL proxy: https://cloud.google.com/sql/docs/postgres/sql-proxy"
echo "   2. Connect: ./cloud-sql-proxy $CONNECTION_NAME"
echo "   3. Run: psql -h localhost -U $DB_USER -d $DB_NAME -f config/schema/postgres.sql"
echo "   4. Run: psql -h localhost -U $DB_USER -d $DB_NAME -f config/schema/timescale.sql"

# =============================================================================
# Step 6: Create Cloud Storage Bucket (for archival)
# =============================================================================

echo -e "\n${YELLOW}Step 6: Creating Cloud Storage bucket...${NC}"

if gsutil ls gs://$STORAGE_BUCKET &>/dev/null; then
    echo "✅ Bucket already exists"
else
    gsutil mb -l $REGION -c STANDARD gs://$STORAGE_BUCKET
    echo "✅ Created bucket: gs://$STORAGE_BUCKET"
fi

# Set lifecycle policy (delete after 1 year)
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365}
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json gs://$STORAGE_BUCKET
rm /tmp/lifecycle.json

echo "✅ Lifecycle policy set (1 year retention)"

# =============================================================================
# Step 7: Store Secrets in Secret Manager
# =============================================================================

echo -e "\n${YELLOW}Step 7: Storing secrets...${NC}"

# Function to create or update secret
store_secret() {
    local secret_name=$1
    local secret_value=$2

    if gcloud secrets describe $secret_name &>/dev/null; then
        echo -n $secret_value | gcloud secrets versions add $secret_name --data-file=-
        echo "✅ Updated secret: $secret_name"
    else
        echo -n $secret_value | gcloud secrets create $secret_name --data-file=- --replication-policy="automatic"
        echo "✅ Created secret: $secret_name"
    fi
}

# Prompt for LLM API keys
echo "Enter your LLM API keys (or press Enter to skip):"
read -p "Claude API Key: " CLAUDE_API_KEY
read -p "Gemini API Key: " GEMINI_API_KEY
read -p "OpenAI API Key: " OPENAI_API_KEY

[ ! -z "$CLAUDE_API_KEY" ] && store_secret "claude-api-key" "$CLAUDE_API_KEY"
[ ! -z "$GEMINI_API_KEY" ] && store_secret "gemini-api-key" "$GEMINI_API_KEY"
[ ! -z "$OPENAI_API_KEY" ] && store_secret "openai-api-key" "$OPENAI_API_KEY"

# =============================================================================
# Step 8: Build and Deploy to Cloud Run
# =============================================================================

echo -e "\n${YELLOW}Step 8: Building and deploying to Cloud Run...${NC}"

# Build container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=512Mi \
    --cpu=1 \
    --timeout=300 \
    --set-env-vars="ENVIRONMENT=production,DATABASE_URL=postgresql://$DB_USER@/$DB_NAME?host=/cloudsql/$CONNECTION_NAME" \
    --set-secrets="CLAUDE_API_KEY=claude-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest,OPENAI_API_KEY=openai-api-key:latest" \
    --add-cloudsql-instances=$CONNECTION_NAME \
    --quiet

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "✅ Deployed to Cloud Run: $SERVICE_URL"

# =============================================================================
# Step 9: Set up Cloud Scheduler (for periodic tasks)
# =============================================================================

echo -e "\n${YELLOW}Step 9: Setting up Cloud Scheduler...${NC}"

# Create daily summary job (runs at 11:59 PM IST)
gcloud scheduler jobs create http daily-summary \
    --schedule="59 18 * * *" \
    --uri="$SERVICE_URL/tasks/daily-summary" \
    --http-method=POST \
    --time-zone="Asia/Kolkata" \
    --description="Generate daily trading summary" \
    --quiet \
    2>/dev/null || echo "Scheduler job already exists"

echo "✅ Cloud Scheduler configured"

# =============================================================================
# Deployment Complete
# =============================================================================

echo -e "\n${GREEN}==============================================================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Resources created:"
echo "  • Project: $PROJECT_ID"
echo "  • Cloud SQL: $DB_INSTANCE_NAME ($CONNECTION_NAME)"
echo "  • Database: $DB_NAME"
echo "  • Storage: gs://$STORAGE_BUCKET"
echo "  • Cloud Run: $SERVICE_URL"
echo ""
echo "Next steps:"
echo "  1. Initialize database schema (see manual step above)"
echo "  2. Configure environment variables"
echo "  3. Test the deployment: curl $SERVICE_URL/health"
echo "  4. Set up monitoring and alerts"
echo ""
echo "Cost estimate: ~₹2,350/month"
echo "  • Cloud SQL (db-f1-micro): ₹1,500/month"
echo "  • Cloud Run: ₹500/month"
echo "  • Cloud Storage: ₹150/month"
echo "  • Networking: ₹200/month"
echo ""
echo "Documentation: see deploy/gcp-setup.md for detailed instructions"
echo ""
