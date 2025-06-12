#!/usr/bin/env python3
"""
GitHub Technical Debt Mining Script
Comprehensive data collection for PhD research on technical debt in AI/ML systems

Requirements:
pip install requests pandas python-dotenv tqdm PyGithub

Usage:
python github_td_miner.py --token YOUR_GITHUB_TOKEN --output_dir td_data --max_repos 100
"""

import os
import json
import time
import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import re
from collections import Counter
import logging
from tqdm import tqdm
import csv

# Optional: More advanced GitHub API
try:
    from github import Github, RateLimitExceededException
    PYGITHUB_AVAILABLE = True
except ImportError:
    print("PyGithub not available. Install with: pip install PyGithub")
    PYGITHUB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubTechnicalDebtMiner:
    def __init__(self, github_token: str, output_dir: str = "td_data"):
        self.token = github_token
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.rate_limit_remaining = 5000
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Technical debt labels to search for
        self.td_labels = [
            'tech-debt', 'technical-debt', 'techdebt', 'tech_debt',
            'refactor', 'refactoring', 'cleanup', 'code-smell',
            'improvement', 'optimize', 'optimization', 'performance',
            'legacy', 'deprecated', 'hack', 'workaround', 'todo',
            'fixme', 'debt', 'technical-debt-management'
        ]
        
        # AI/ML related keywords for filtering repositories
        self.ai_ml_keywords = [
            'machine learning', 'deep learning', 'artificial intelligence',
            'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas',
            'numpy', 'transformers', 'huggingface', 'bert', 'gpt',
            'neural network', 'computer vision', 'nlp', 'natural language',
            'reinforcement learning', 'data science', 'ml', 'ai'
        ]
        
        # Initialize PyGithub if available
        self.github_client = Github(github_token) if PYGITHUB_AVAILABLE else None

    def search_ai_ml_repositories(self, max_repos: int = 200, min_stars: int = 50) -> List[Dict]:
        """
        Search for AI/ML repositories with good activity and potential technical debt
        """
        logger.info("Searching for AI/ML repositories...")
        
        repositories = []
        
        # Search queries targeting different AI/ML domains
        search_queries = [
            f'language:python "machine learning" stars:>={min_stars}',
            f'language:python "deep learning" stars:>={min_stars}',
            f'language:python pytorch stars:>={min_stars}',
            f'language:python tensorflow stars:>={min_stars}',
            f'language:python transformers huggingface stars:>={min_stars}',
            f'language:python "computer vision" opencv stars:>={min_stars}',
            f'language:python "reinforcement learning" stars:>={min_stars}',
            f'language:python scikit-learn "data science" stars:>={min_stars}',
            f'language:python keras "neural network" stars:>={min_stars}',
            f'language:python pandas numpy "data analysis" stars:>={min_stars}'
        ]
        
        repos_per_query = max_repos // len(search_queries)
        
        for query in search_queries:
            try:
                query_repos = self._search_repositories(query, max_results=repos_per_query)
                repositories.extend(query_repos)
                logger.info(f"Found {len(query_repos)} repositories for query: {query[:50]}...")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching with query '{query}': {e}")
                continue
        
        # Remove duplicates
        unique_repos = {}
        for repo in repositories:
            if repo['full_name'] not in unique_repos:
                unique_repos[repo['full_name']] = repo
        
        final_repos = list(unique_repos.values())[:max_repos]
        logger.info(f"Total unique AI/ML repositories found: {len(final_repos)}")
        
        return final_repos

    def _search_repositories(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search repositories using GitHub API"""
        repositories = []
        page = 1
        per_page = min(100, max_results)
        
        while len(repositories) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': per_page,
                'page': page
            }
            
            response = self._make_request(url, params)
            if not response or 'items' not in response:
                break
            
            items = response['items']
            if not items:
                break
            
            for repo in items:
                if self._is_relevant_ai_ml_repo(repo):
                    repo_data = self._extract_repo_metadata(repo)
                    repositories.append(repo_data)
            
            page += 1
            if len(items) < per_page:  # Last page
                break
        
        return repositories[:max_results]

    def _is_relevant_ai_ml_repo(self, repo: Dict) -> bool:
        """Check if repository is relevant for AI/ML technical debt research"""
        # Basic filters
        if repo.get('private', True):
            return False
        if repo.get('archived', False):
            return False
        if repo.get('language') != 'Python':
            return False
        if repo.get('stargazers_count', 0) < 10:
            return False
        
        # Check if repo contains AI/ML keywords
        text_to_check = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
        return any(keyword in text_to_check for keyword in self.ai_ml_keywords)

    def _extract_repo_metadata(self, repo: Dict) -> Dict:
        """Extract relevant metadata from repository"""
        return {
            'full_name': repo['full_name'],
            'name': repo['name'],
            'owner': repo['owner']['login'],
            'description': repo.get('description', ''),
            'language': repo.get('language'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'issues_count': repo.get('open_issues_count', 0),
            'created_at': repo.get('created_at'),
            'updated_at': repo.get('updated_at'),
            'size': repo.get('size', 0),
            'has_issues': repo.get('has_issues', False),
            'has_wiki': repo.get('has_wiki', False),
            'url': repo['html_url'],
            'clone_url': repo['clone_url']
        }

    def collect_technical_debt_issues(self, repositories: List[Dict]) -> List[Dict]:
        """
        Collect technical debt issues from repositories
        """
        logger.info(f"Collecting technical debt issues from {len(repositories)} repositories...")
        
        all_issues = []
        
        for i, repo in enumerate(tqdm(repositories, desc="Processing repositories")):
            repo_name = repo['full_name']
            logger.info(f"Processing {repo_name} ({i+1}/{len(repositories)})")
            
            try:
                # Method 1: Search by labels
                labeled_issues = self._get_issues_by_labels(repo_name)
                
                # Method 2: Search by content
                content_issues = self._search_issues_by_content(repo_name)
                
                # Combine and deduplicate
                repo_issues = self._deduplicate_issues(labeled_issues + content_issues)
                
                # Enhance issues with additional data
                enhanced_issues = []
                for issue in repo_issues:
                    enhanced_issue = self._enhance_issue_data(issue, repo)
                    enhanced_issues.append(enhanced_issue)
                
                all_issues.extend(enhanced_issues)
                logger.info(f"Found {len(enhanced_issues)} TD issues in {repo_name}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {repo_name}: {e}")
                continue
        
        logger.info(f"Total technical debt issues collected: {len(all_issues)}")
        return all_issues

    def _get_issues_by_labels(self, repo_name: str) -> List[Dict]:
        """Get issues with technical debt labels"""
        issues = []
        
        # Search for issues with TD labels
        for label in self.td_labels[:5]:  # Limit to avoid rate limits
            url = f"{self.base_url}/repos/{repo_name}/issues"
            params = {
                'labels': label,
                'state': 'all',
                'per_page': 100
            }
            
            response = self._make_request(url, params)
            if response and isinstance(response, list):
                issues.extend(response)
        
        return issues

    def _search_issues_by_content(self, repo_name: str) -> List[Dict]:
        """Search issues by content containing TD keywords"""
        issues = []
        
        # Search for specific technical debt phrases
        td_phrases = [
            'technical debt', 'refactor', 'cleanup', 'code smell',
            'legacy code', 'hack', 'workaround'
        ]
        
        for phrase in td_phrases[:3]:  # Limit to avoid rate limits
            query = f'repo:{repo_name} is:issue "{phrase}"'
            url = f"{self.base_url}/search/issues"
            params = {
                'q': query,
                'per_page': 50
            }
            
            response = self._make_request(url, params)
            if response and 'items' in response:
                issues.extend(response['items'])
        
        return issues

    def _deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Remove duplicate issues"""
        seen_issues = set()
        unique_issues = []
        
        for issue in issues:
            issue_id = issue.get('id')
            if issue_id and issue_id not in seen_issues:
                seen_issues.add(issue_id)
                unique_issues.append(issue)
        
        return unique_issues

    def _enhance_issue_data(self, issue: Dict, repo: Dict) -> Dict:
        """Enhance issue with additional metadata and analysis"""
        enhanced = {
            # Basic issue info
            'repository': repo['full_name'],
            'repo_stars': repo['stars'],
            'repo_language': repo['language'],
            'issue_id': issue['id'],
            'issue_number': issue['number'],
            'title': issue['title'],
            'body': issue.get('body', ''),
            'state': issue['state'],
            'created_at': issue['created_at'],
            'updated_at': issue['updated_at'],
            'closed_at': issue.get('closed_at'),
            'author': issue['user']['login'],
            'labels': [label['name'] for label in issue.get('labels', [])],
            'comments_count': issue.get('comments', 0),
            'assignees': [a['login'] for a in issue.get('assignees', [])],
            'url': issue['html_url'],
            
            # Technical debt analysis
            'td_classification': self._classify_technical_debt(issue),
            'td_severity': self._assess_td_severity(issue),
            'ai_ml_context': self._identify_ai_ml_context(issue, repo),
            'resolution_time_days': self._calculate_resolution_time(issue),
            'is_valid_td': self._validate_technical_debt(issue)
        }
        
        return enhanced

    def _classify_technical_debt(self, issue: Dict) -> Dict:
        """Classify the type of technical debt"""
        title_body = f"{issue['title']} {issue.get('body', '')}".lower()
        
        td_categories = {
            'code_debt': [
                'refactor', 'cleanup', 'code smell', 'duplicate code',
                'complex function', 'long method', 'god class'
            ],
            'design_debt': [
                'architecture', 'design pattern', 'coupling', 'cohesion',
                'dependency', 'interface'
            ],
            'documentation_debt': [
                'documentation', 'readme', 'comment', 'docstring',
                'api doc', 'user guide'
            ],
            'test_debt': [
                'test', 'unit test', 'integration test', 'coverage',
                'testing', 'mock'
            ],
            'performance_debt': [
                'performance', 'optimization', 'slow', 'memory',
                'cpu', 'bottleneck', 'efficiency'
            ],
            'security_debt': [
                'security', 'vulnerability', 'authentication',
                'authorization', 'encryption'
            ],
            'data_debt': [
                'data quality', 'data pipeline', 'data validation',
                'dataset', 'preprocessing', 'feature engineering'
            ],
            'model_debt': [
                'model', 'training', 'accuracy', 'overfitting',
                'model drift', 'hyperparameter', 'evaluation'
            ],
            'infrastructure_debt': [
                'deployment', 'scaling', 'docker', 'kubernetes',
                'cloud', 'infrastructure', 'devops'
            ]
        }
        
        classification = {}
        for category, keywords in td_categories.items():
            score = sum(1 for keyword in keywords if keyword in title_body)
            if score > 0:
                classification[category] = score
        
        # Determine primary category
        primary_category = max(classification.keys(), key=classification.get) if classification else 'general'
        
        return {
            'primary_category': primary_category,
            'all_categories': classification,
            'is_ai_ml_specific': primary_category in ['data_debt', 'model_debt']
        }

    def _assess_td_severity(self, issue: Dict) -> str:
        """Assess technical debt severity"""
        title_body = f"{issue['title']} {issue.get('body', '')}".lower()
        
        high_severity_indicators = [
            'critical', 'urgent', 'blocking', 'production', 'crash',
            'security', 'memory leak', 'performance issue'
        ]
        
        medium_severity_indicators = [
            'important', 'should fix', 'improvement', 'optimization',
            'refactor needed', 'slow'
        ]
        
        if any(indicator in title_body for indicator in high_severity_indicators):
            return 'high'
        elif any(indicator in title_body for indicator in medium_severity_indicators):
            return 'medium'
        else:
            return 'low'

    def _identify_ai_ml_context(self, issue: Dict, repo: Dict) -> List[str]:
        """Identify AI/ML context of the issue"""
        text = f"{issue['title']} {issue.get('body', '')} {repo['description']}".lower()
        
        contexts = {
            'machine_learning': ['ml', 'machine learning', 'model', 'training', 'prediction'],
            'deep_learning': ['deep learning', 'neural network', 'pytorch', 'tensorflow', 'keras'],
            'nlp': ['nlp', 'natural language', 'text', 'bert', 'gpt', 'transformers'],
            'computer_vision': ['computer vision', 'cv', 'image', 'opencv', 'vision'],
            'data_science': ['data science', 'pandas', 'numpy', 'jupyter', 'notebook'],
            'reinforcement_learning': ['reinforcement learning', 'rl', 'agent', 'environment']
        }
        
        identified_contexts = []
        for context, keywords in contexts.items():
            if any(keyword in text for keyword in keywords):
                identified_contexts.append(context)
        
        return identified_contexts

    def _calculate_resolution_time(self, issue: Dict) -> Optional[int]:
        """Calculate time to resolve issue in days"""
        if issue['state'] != 'closed' or not issue.get('closed_at'):
            return None
        
        created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
        closed = datetime.fromisoformat(issue['closed_at'].replace('Z', '+00:00'))
        return (closed - created).days

    def _validate_technical_debt(self, issue: Dict) -> bool:
        """Validate that issue actually represents technical debt"""
        title_body = f"{issue['title']} {issue.get('body', '')}".lower()
        
        # Positive indicators
        positive_keywords = [
            'refactor', 'cleanup', 'improve', 'optimize', 'debt',
            'smell', 'legacy', 'hack', 'workaround', 'technical debt'
        ]
        
        # Negative indicators (likely false positives)
        negative_keywords = [
            'new feature', 'enhancement request', 'question',
            'documentation update', 'version bump'
        ]
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in title_body)
        negative_score = sum(1 for keyword in negative_keywords if keyword in title_body)
        
        return positive_score > 0 and positive_score > negative_score

    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        if self.rate_limit_remaining < 100:
            logger.info("Rate limit low, sleeping for 60 seconds...")
            time.sleep(60)
            self.rate_limit_remaining = 5000
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            
            if response.status_code == 403:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                sleep_time = reset_time - int(time.time()) + 1
                if sleep_time > 0:
                    logger.info(f"Rate limit exceeded. Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    return self._make_request(url, params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def analyze_collected_data(self, issues: List[Dict]) -> Dict:
        """Analyze collected technical debt data"""
        if not issues:
            return {'error': 'No data to analyze'}
        
        analysis = {
            'total_issues': len(issues),
            'total_repositories': len(set(issue['repository'] for issue in issues)),
            'issues_by_state': Counter(issue['state'] for issue in issues),
            'issues_by_severity': Counter(issue['td_severity'] for issue in issues),
            'issues_by_primary_category': Counter(
                issue['td_classification']['primary_category'] for issue in issues
            ),
            'ai_ml_contexts': Counter([
                context for issue in issues 
                for context in issue['ai_ml_context']
            ]),
            'avg_resolution_time': self._calculate_avg_resolution_time(issues),
            'repositories_with_most_td': Counter(
                issue['repository'] for issue in issues
            ).most_common(10),
            'temporal_trends': self._analyze_temporal_trends(issues)
        }
        
        return analysis

    def _calculate_avg_resolution_time(self, issues: List[Dict]) -> float:
        """Calculate average resolution time"""
        resolution_times = [
            issue['resolution_time_days'] for issue in issues 
            if issue['resolution_time_days'] is not None
        ]
        return sum(resolution_times) / len(resolution_times) if resolution_times else 0

    def _analyze_temporal_trends(self, issues: List[Dict]) -> Dict:
        """Analyze temporal trends in technical debt"""
        # Group issues by month
        monthly_counts = Counter()
        
        for issue in issues:
            created_date = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
            month_key = created_date.strftime('%Y-%m')
            monthly_counts[month_key] += 1
        
        return dict(monthly_counts.most_common(12))

    def save_data(self, repositories: List[Dict], issues: List[Dict], analysis: Dict):
        """Save all collected data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save repositories
        repos_file = os.path.join(self.output_dir, f'repositories_{timestamp}.json')
        with open(repos_file, 'w') as f:
            json.dump(repositories, f, indent=2, default=str)
        
        repos_csv = os.path.join(self.output_dir, f'repositories_{timestamp}.csv')
        pd.DataFrame(repositories).to_csv(repos_csv, index=False)
        
        # Save issues
        issues_file = os.path.join(self.output_dir, f'issues_{timestamp}.json')
        with open(issues_file, 'w') as f:
            json.dump(issues, f, indent=2, default=str)
        
        issues_csv = os.path.join(self.output_dir, f'issues_{timestamp}.csv')
        pd.DataFrame(issues).to_csv(issues_csv, index=False)
        
        # Save analysis
        analysis_file = os.path.join(self.output_dir, f'analysis_{timestamp}.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(repositories, issues, analysis, timestamp)
        
        logger.info(f"Data saved to {self.output_dir}")

    def _generate_summary_report(self, repositories: List[Dict], issues: List[Dict], 
                                analysis: Dict, timestamp: str):
        """Generate human-readable summary report"""
        
        report = f"""
TECHNICAL DEBT MINING REPORT
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REPOSITORIES ANALYZED
--------------------
Total repositories: {len(repositories)}
Average stars: {sum(r['stars'] for r in repositories) / len(repositories):.1f}
Languages: {Counter(r['language'] for r in repositories)}

TECHNICAL DEBT ISSUES
--------------------
Total issues found: {analysis['total_issues']}
Valid TD issues: {len([i for i in issues if i['is_valid_td']])}
Average resolution time: {analysis['avg_resolution_time']:.1f} days

ISSUE BREAKDOWN
--------------
By State: {dict(analysis['issues_by_state'])}
By Severity: {dict(analysis['issues_by_severity'])}
By Category: {dict(analysis['issues_by_primary_category'])}

AI/ML CONTEXTS
--------------
{dict(analysis['ai_ml_contexts'])}

TOP REPOSITORIES WITH TECHNICAL DEBT
-----------------------------------
"""
        
        for repo, count in analysis['repositories_with_most_td']:
            report += f"{repo}: {count} issues\n"
        
        report_file = os.path.join(self.output_dir, f'summary_report_{timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)

def main():
    parser = argparse.ArgumentParser(description='Mine GitHub repositories for technical debt data')
    parser.add_argument('--token', required=True, help='GitHub API token')
    parser.add_argument('--output_dir', default='td_data', help='Output directory')
    parser.add_argument('--max_repos', type=int, default=50, help='Maximum repositories to analyze')
    parser.add_argument('--min_stars', type=int, default=50, help='Minimum stars for repositories')
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = GitHubTechnicalDebtMiner(args.token, args.output_dir)
    
    try:
        # Step 1: Find AI/ML repositories
        repositories = miner.search_ai_ml_repositories(
            max_repos=args.max_repos,
            min_stars=args.min_stars
        )
        
        if not repositories:
            logger.error("No repositories found. Check your token and criteria.")
            return
        
        # Step 2: Collect technical debt issues
        issues = miner.collect_technical_debt_issues(repositories)
        
        # Step 3: Analyze data
        analysis = miner.analyze_collected_data(issues)
        
        # Step 4: Save everything
        miner.save_data(repositories, issues, analysis)
        
        logger.info("Technical debt mining completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Mining interrupted by user")
    except Exception as e:
        logger.error(f"Error during mining: {e}")
        raise

if __name__ == "__main__":
    main()