#!/bin/bash
# Deploy Aletheia API to logos
# Run from the project root on your local machine (penguin)

set -e

REMOTE_HOST="10.0.0.1"
REMOTE_USER="chapinad"
REMOTE_DIR="/home/chapinad/aletheia-api"

echo "=== Deploying Aletheia API to logos @ ${REMOTE_HOST} ==="
echo ""

echo "1. Creating remote directory..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}"

echo ""
echo "2. Creating temporary archive..."
tar czf /tmp/aletheia-deploy.tar.gz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='htmlcov' \
    --exclude='.coverage' \
    --exclude='.env' \
    .

echo ""
echo "3. Copying archive to logos..."
scp /tmp/aletheia-deploy.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:/tmp/

echo ""
echo "4. Extracting on remote..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && tar xzf /tmp/aletheia-deploy.tar.gz && rm /tmp/aletheia-deploy.tar.gz"

echo ""
echo "5. Cleaning up local archive..."
rm /tmp/aletheia-deploy.tar.gz

echo ""
echo "6. Setting up .env file..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && cp .env.prod .env"

echo ""
echo "=== Files transferred! ==="
echo ""
echo "Next steps (run on logos):"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  # Edit .env and add your OpenAI API key"
echo "  docker compose -f docker-compose.prod.yml build"
echo "  docker compose -f docker-compose.prod.yml up -d"
echo "  docker compose -f docker-compose.prod.yml logs -f"
echo ""
