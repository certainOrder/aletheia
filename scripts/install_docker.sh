#!/bin/bash
# Install Docker on Debian (logos)
# Run this script on logos server

set -e

echo "=== Installing Docker on Debian ==="
echo ""

echo "1. Update package index..."
sudo apt-get update

echo ""
echo "2. Install prerequisites..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo ""
echo "3. Add Docker's official GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo ""
echo "4. Set up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo ""
echo "5. Update package index with Docker repo..."
sudo apt-get update

echo ""
echo "6. Install Docker Engine, CLI, and Docker Compose..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo ""
echo "7. Add current user to docker group (to run without sudo)..."
sudo usermod -aG docker $USER

echo ""
echo "8. Enable and start Docker service..."
sudo systemctl enable docker
sudo systemctl start docker

echo ""
echo "9. Verify installation..."
sudo docker --version
sudo docker compose version

echo ""
echo "=== Docker Installation Complete! ==="
echo ""
echo "⚠️  IMPORTANT: You need to log out and back in for group changes to take effect."
echo "    Or run: newgrp docker"
echo ""
echo "Test with: docker run hello-world"
