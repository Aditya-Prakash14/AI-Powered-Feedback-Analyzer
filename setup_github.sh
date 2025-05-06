#!/bin/bash

# This script helps initialize a Git repository and push it to GitHub
# Usage: ./setup_github.sh <github_repo_url>

# Check if a GitHub URL was provided
if [ -z "$1" ]; then
    echo "Please provide a GitHub repository URL."
    echo "Usage: ./setup_github.sh <github_repo_url>"
    exit 1
fi

GITHUB_URL=$1

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Add all files
echo "Adding files to Git..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit of Product Pulse: AI-Powered Feedback Analyzer"

# Add GitHub remote
echo "Adding GitHub remote..."
git remote add origin $GITHUB_URL

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin master || git push -u origin main

echo "Done! Your code has been pushed to GitHub."
echo "Repository URL: $GITHUB_URL"
