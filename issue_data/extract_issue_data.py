#!/usr/bin/env python3
"""
GitHub Issue Data Extractor for Technical Debt Research in AI/ML Services
Search for relevant repositories based on carefully crafted queries for CV, NLP, and RL services.


This script automates the process of finding relevant GitHub repositories
and extracting their full issue data (including titles, bodies, and comments)
for qualitative analysis.

It is designed to find repositories that build services or applications on
top of cloud ML APIs and LLMs.

Usage:
1. Install dependencies:
   pip install requests tqdm

2. Get a GitHub Personal Access Token:
   Go to https://github.com/settings/tokens and generate a new token
   with `public_repo` scope.

3. Run the script:
   python extract_issue_data.py --token YOUR_GITHUB_TOKEN --output_dir ./issue_data
"""

import os
import json
import time
import requests
import argparse
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IssueDataExtractor:
    def __init__(self, github_token: str, output_dir: str):
        if not github_token:
            raise ValueError("GitHub token cannot be empty.")
        self.token = github_token
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Search queries to find relevant repositories ---
        # These are designed to find APPLICATIONS and SERVICES, not just libraries.
        # You can customize these to broaden or narrow your search.
        self.search_queries = {
            "NLP/LLM_Services": [
                'langchain app "deploy" language:python',
                'quivr OR "privategpt" OR "anything-llm" language:python',
                '"openai api" "streamlit app" language:python',
                '"aws bedrock" "service" "pipeline" language:python',
                '"azure openai" "service" "fastapi" language:python'
            ],
            "Computer_Vision_Services": [
                '"google cloud vision" "api" "app" language:python',
                '"aws rekognition" "pipeline" "service" language:python',
                '"openai vision api" "service" "app" language:python',
                'roboflow supervision language:python',
                'computervision "edge" "azure" "deployment" language:python'
            ],
            "RL_Services": [
                '"reinforcement learning" "aws sagemaker" "deployment" language:python',
                '"rllib" "service" "deploy" language:python',
                '"huggingface lerobot" "robotics" language:python'
            ]
        }

    def _make_request(self, url: str, params: dict = None, a_headers: dict = None) -> dict or list or None:
        """Makes a request to the GitHub API with rate limit handling."""
        try:
            response = requests.get(url, headers=a_headers or self.headers, params=params)

            # Check for rate limiting
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                if remaining < 50:
                    reset_time = int(response.headers['X-RateLimit-Reset'])
                    sleep_duration = max(reset_time - time.time(), 1)
                    logger.warning(f"Rate limit low ({remaining}). Sleeping for {sleep_duration:.0f} seconds.")
                    time.sleep(sleep_duration)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.error("403 Forbidden Error. Check your token permissions or rate limits.")
            else:
                logger.error(f"HTTP Error for URL {url}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL {url}: {e}")
        return None

    def search_repositories(self, max_repos_per_query: int = 5) -> set:
        """Searches GitHub for relevant repositories based on the defined queries."""
        logger.info("Starting repository search...")
        unique_repo_names = set()

        for category, queries in self.search_queries.items():
            logger.info(f"--- Searching in category: {category} ---")
            for query in queries:
                # Add filters for quality: recent activity and some popularity
                full_query = f'{query} stars:>50 pushed:>{(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")}'
                params = {
                    'q': full_query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': max_repos_per_query
                }
                logger.info(f"Executing query: {full_query}")
                results = self._make_request(f"{self.base_url}/search/repositories", params)

                if results and 'items' in results:
                    for repo in results['items']:
                        repo_name = repo['full_name']
                        logger.info(f"  Found repository: {repo_name} ({repo['stargazers_count']} stars)")
                        unique_repo_names.add(repo_name)
                time.sleep(5)  # Be respectful to the API between searches

        logger.info(f"\nFound a total of {len(unique_repo_names)} unique repositories for analysis.")
        return unique_repo_names

    def fetch_issues_for_repo(self, repo_name: str, max_issues: int = 200):
        """Fetches all issue data for a single repository, including comments."""
        logger.info(f"Fetching issues for {repo_name}...")
        issues_with_comments = []
        
        # Fetch issues updated in the last 2 years to keep data relevant
        since_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        params = {
            'state': 'all',  # Get both open and closed issues
            'since': since_date,
            'per_page': 100, # Max per page
            'direction': 'desc',
            'sort': 'updated'
        }
        
        url = f"{self.base_url}/repos/{repo_name}/issues"
        
        page = 1
        while True:
            params['page'] = page
            issues = self._make_request(url, params)
            if not issues:
                break
                
            for issue in tqdm(issues, desc=f"Processing issues for {repo_name} (Page {page})"):
                # Skip pull requests, which are also returned by the issues API
                if 'pull_request' in issue:
                    continue

                # Fetch comments for this specific issue
                issue_comments = []
                if issue.get('comments', 0) > 0:
                    comments_data = self._make_request(issue['comments_url'])
                    if comments_data:
                        issue_comments = [{'user': c['user']['login'], 'body': c['body']} for c in comments_data]

                # Consolidate all data for this issue
                issues_with_comments.append({
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'author': issue['user']['login'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'labels': [label['name'] for label in issue['labels']],
                    'body': issue['body'],
                    'comments': issue_comments
                })

                if len(issues_with_comments) >= max_issues:
                    break
            
            if len(issues_with_comments) >= max_issues or len(issues) < 100:
                break
            
            page += 1
            time.sleep(1) # Pause between pages

        logger.info(f"Extracted {len(issues_with_comments)} issues from {repo_name}.")
        return issues_with_comments

    def run_extraction(self):
        """Orchestrates the entire search and extraction process."""
        # 1. Find repositories
        repos_to_analyze = self.search_repositories()
        
        if not repos_to_analyze:
            logger.warning("No repositories found. Try adjusting your search queries.")
            return

        # 2. Extract issue data for each repository
        for repo_name in repos_to_analyze:
            # Sanitize repo name for use as a filename
            safe_filename = repo_name.replace('/', '_') + '_issues.json'
            output_path = os.path.join(self.output_dir, safe_filename)

            if os.path.exists(output_path):
                logger.info(f"Skipping {repo_name}, data already exists at {output_path}")
                continue

            repo_issues = self.fetch_issues_for_repo(repo_name)
            
            if repo_issues:
                # 3. Save the results for the current repo
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(repo_issues, f, indent=2, ensure_ascii=False)
                logger.info(f"Successfully saved data for {repo_name} to {output_path}")
            
            # Pause between processing repositories
            time.sleep(10)

        logger.info("\nExtraction process completed!")

def main():
    parser = argparse.ArgumentParser(description='GitHub Issue Data Extractor for TD Research in AI/ML Services.')
    parser.add_argument('--token', required=True, help='Your GitHub Personal Access Token.')
    # This is the corrected line
    parser.add_argument('--output_dir', default='./issue_data', help='Directory to save the output JSON files.')
    
    args = parser.parse_args()
    
    extractor = IssueDataExtractor(github_token=args.token, output_dir=args.output_dir)
    extractor.run_extraction()

if __name__ == "__main__":
    main()