#!/usr/bin/env python3
"""
Discover Technical Debt Labels in LLM/AI Service Repositories
Empirical exploration to find what TD/SATD labels actually exist in practice

Usage:
python discover_td_labels.py --token YOUR_GITHUB_TOKEN --output labels_discovery.json
"""

import os
import json
import time
import requests
import pandas as pd
import argparse
from datetime import datetime
from typing import List, Dict, Set
from collections import Counter, defaultdict
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalDebtLabelDiscoverer:
    def __init__(self, github_token: str, output_file: str = "labels_discovery.json"):
        self.token = github_token
        self.output_file = output_file
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.rate_limit_remaining = 5000
        
        # LLM and AI service related repository search terms
        self.llm_ai_search_terms = [
            'openai gpt python',
            'langchain python', 
            'anthropic claude python',
            'aws bedrock python',
            'azure openai python',
            'google vertex ai python',
            'huggingface transformers python',
            'llamaindex python',
            'streamlit openai python',
            'fastapi openai python'
        ]
        
        # Potential technical debt label patterns to look for
        self.potential_td_patterns = [
            'debt', 'refactor', 'cleanup', 'improve', 'optimize',
            'fix', 'hack', 'workaround', 'todo', 'legacy',
            'technical', 'maintenance', 'enhancement', 'performance'
        ]

    def discover_llm_ai_repositories(self, max_repos_per_term: int = 50) -> List[Dict]:
        """
        Discover LLM/AI service repositories
        """
        logger.info("Discovering LLM/AI service repositories...")
        
        all_repositories = []
        
        for search_term in self.llm_ai_search_terms:
            logger.info(f"Searching for repositories with: {search_term}")
            
            try:
                repos = self._search_repositories(search_term, max_repos_per_term)
                all_repositories.extend(repos)
                logger.info(f"Found {len(repos)} repositories for '{search_term}'")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching '{search_term}': {e}")
                continue
        
        # Remove duplicates
        unique_repos = {}
        for repo in all_repositories:
            if repo['full_name'] not in unique_repos:
                unique_repos[repo['full_name']] = repo
        
        final_repos = list(unique_repos.values())
        logger.info(f"Total unique LLM/AI repositories found: {len(final_repos)}")
        
        return final_repos

    def _search_repositories(self, search_term: str, max_results: int) -> List[Dict]:
        """Search repositories for a specific term"""
        repositories = []
        page = 1
        per_page = min(100, max_results)
        
        while len(repositories) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                'q': f'language:python {search_term} stars:>10 pushed:>2023-01-01',
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
                if self._is_relevant_repo(repo):
                    repo_data = {
                        'full_name': repo['full_name'],
                        'name': repo['name'],
                        'description': repo.get('description', ''),
                        'stars': repo.get('stargazers_count', 0),
                        'language': repo.get('language'),
                        'created_at': repo.get('created_at'),
                        'updated_at': repo.get('updated_at'),
                        'has_issues': repo.get('has_issues', False),
                        'search_term': search_term
                    }
                    repositories.append(repo_data)
            
            page += 1
            if len(items) < per_page:
                break
        
        return repositories[:max_results]

    def _is_relevant_repo(self, repo: Dict) -> bool:
        """Check if repository is relevant for our analysis"""
        if repo.get('private', True):
            return False
        if repo.get('archived', False):
            return False
        if repo.get('language') != 'Python':
            return False
        if repo.get('stargazers_count', 0) < 5:
            return False
        if not repo.get('has_issues', False):
            return False
        
        return True

    def discover_all_labels(self, repositories: List[Dict]) -> Dict:
        """
        Discover all labels used in LLM/AI repositories
        """
        logger.info(f"Discovering labels from {len(repositories)} repositories...")
        
        all_labels = Counter()
        label_contexts = defaultdict(list)
        repo_label_stats = []
        
        for i, repo in enumerate(tqdm(repositories, desc="Analyzing repository labels")):
            repo_name = repo['full_name']
            logger.info(f"Analyzing labels in {repo_name} ({i+1}/{len(repositories)})")
            
            try:
                # Get all labels from the repository
                repo_labels = self._get_repository_labels(repo_name)
                
                # Count labels
                for label in repo_labels:
                    label_name = label['name'].lower()
                    all_labels[label_name] += 1
                    
                    # Store context information
                    label_contexts[label_name].append({
                        'repository': repo_name,
                        'description': label.get('description', ''),
                        'color': label.get('color', ''),
                        'repo_stars': repo['stars']
                    })
                
                # Store repository statistics
                repo_stats = {
                    'repository': repo_name,
                    'total_labels': len(repo_labels),
                    'stars': repo['stars'],
                    'search_term': repo.get('search_term', ''),
                    'labels': [label['name'] for label in repo_labels]
                }
                repo_label_stats.append(repo_stats)
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error analyzing {repo_name}: {e}")
                continue
        
        logger.info(f"Discovered {len(all_labels)} unique labels")
        
        return {
            'label_frequency': dict(all_labels),
            'label_contexts': dict(label_contexts),
            'repository_stats': repo_label_stats
        }

    def _get_repository_labels(self, repo_name: str) -> List[Dict]:
        """Get all labels from a repository"""
        url = f"{self.base_url}/repos/{repo_name}/labels"
        params = {'per_page': 100}
        
        response = self._make_request(url, params)
        if response and isinstance(response, list):
            return response
        return []

    def identify_technical_debt_labels(self, all_labels_data: Dict) -> Dict:
        """
        Identify which labels are related to technical debt
        """
        logger.info("Identifying technical debt related labels...")
        
        label_frequency = all_labels_data['label_frequency']
        label_contexts = all_labels_data['label_contexts']
        
        # Find potential TD labels
        potential_td_labels = {}
        definite_td_labels = {}
        
        for label_name, frequency in label_frequency.items():
            # Check if label contains TD patterns
            td_score = 0
            matched_patterns = []
            
            for pattern in self.potential_td_patterns:
                if pattern in label_name:
                    td_score += 1
                    matched_patterns.append(pattern)
            
            # Analyze label descriptions for additional context
            descriptions = [ctx.get('description', '') or '' for ctx in label_contexts[label_name]]
            description_text = ' '.join(descriptions).lower()
            
            for pattern in self.potential_td_patterns:
                if pattern in description_text:
                    td_score += 0.5
                    if pattern not in matched_patterns:
                        matched_patterns.append(f"{pattern} (in description)")
            
            # Categorize labels
            if td_score >= 1:
                definite_td_labels[label_name] = {
                    'frequency': frequency,
                    'td_score': td_score,
                    'matched_patterns': matched_patterns,
                    'sample_contexts': label_contexts[label_name][:3],  # Top 3 examples
                    'avg_repo_stars': sum(ctx['repo_stars'] for ctx in label_contexts[label_name]) / frequency
                }
            elif td_score > 0:
                potential_td_labels[label_name] = {
                    'frequency': frequency,
                    'td_score': td_score,
                    'matched_patterns': matched_patterns,
                    'sample_contexts': label_contexts[label_name][:3]
                }
        
        return {
            'definite_td_labels': definite_td_labels,
            'potential_td_labels': potential_td_labels,
            'total_labels_analyzed': len(label_frequency)
        }

    def analyze_llm_specific_patterns(self, all_labels_data: Dict) -> Dict:
        """
        Look for LLM/AI specific patterns in labels
        """
        logger.info("Analyzing LLM/AI specific label patterns...")
        
        llm_ai_patterns = [
            'api', 'prompt', 'model', 'llm', 'openai', 'gpt', 'claude',
            'embeddings', 'tokens', 'context', 'chat', 'completion',
            'fine-tune', 'training', 'inference', 'deployment'
        ]
        
        label_frequency = all_labels_data['label_frequency']
        label_contexts = all_labels_data['label_contexts']
        
        llm_related_labels = {}
        
        for label_name, frequency in label_frequency.items():
            llm_score = 0
            matched_llm_patterns = []
            
            for pattern in llm_ai_patterns:
                if pattern in label_name:
                    llm_score += 1
                    matched_llm_patterns.append(pattern)
            
            # Check descriptions too
            descriptions = [ctx.get('description', '') or '' for ctx in label_contexts[label_name]]
            description_text = ' '.join(descriptions).lower()
            
            for pattern in llm_ai_patterns:
                if pattern in description_text:
                    llm_score += 0.5
                    if pattern not in matched_llm_patterns:
                        matched_llm_patterns.append(f"{pattern} (in description)")
            
            if llm_score > 0:
                llm_related_labels[label_name] = {
                    'frequency': frequency,
                    'llm_score': llm_score,
                    'matched_patterns': matched_llm_patterns,
                    'sample_contexts': label_contexts[label_name][:2]
                }
        
        return llm_related_labels

    def generate_comprehensive_report(self, repositories: List[Dict], all_labels_data: Dict, 
                                    td_analysis: Dict, llm_analysis: Dict) -> Dict:
        """
        Generate comprehensive analysis report
        """
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_repositories_analyzed': len(repositories),
                'total_unique_labels_found': len(all_labels_data['label_frequency']),
                'search_terms_used': self.llm_ai_search_terms
            },
            
            'repository_summary': {
                'repositories_by_search_term': Counter(repo.get('search_term', 'unknown') for repo in repositories),
                'avg_stars': sum(repo['stars'] for repo in repositories) / len(repositories),
                'repositories_with_labels': len([r for r in all_labels_data['repository_stats'] if r['total_labels'] > 0]),
                'avg_labels_per_repo': sum(r['total_labels'] for r in all_labels_data['repository_stats']) / len(all_labels_data['repository_stats'])
            },
            
            'technical_debt_findings': {
                'definite_td_labels_count': len(td_analysis['definite_td_labels']),
                'potential_td_labels_count': len(td_analysis['potential_td_labels']),
                'most_common_td_labels': sorted(td_analysis['definite_td_labels'].items(), 
                                              key=lambda x: x[1]['frequency'], reverse=True)[:10],
                'td_adoption_rate': len(td_analysis['definite_td_labels']) / len(all_labels_data['label_frequency']) * 100
            },
            
            'llm_ai_findings': {
                'llm_related_labels_count': len(llm_analysis),
                'most_common_llm_labels': sorted(llm_analysis.items(), 
                                                key=lambda x: x[1]['frequency'], reverse=True)[:10]
            },
            
            'detailed_data': {
                'repositories': repositories,
                'all_labels': all_labels_data,
                'td_analysis': td_analysis,
                'llm_analysis': llm_analysis
            }
        }
        
        return report

    def _make_request(self, url: str, params: Dict = None) -> Dict:
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

    def save_results(self, report: Dict):
        """Save analysis results"""
        # Save full report
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save CSV summary for easy analysis
        csv_file = self.output_file.replace('.json', '_summary.csv')
        
        # Create TD labels summary
        td_labels_data = []
        for label_name, data in report['detailed_data']['td_analysis']['definite_td_labels'].items():
            td_labels_data.append({
                'label_name': label_name,
                'frequency': data['frequency'],
                'td_score': data['td_score'],
                'matched_patterns': ', '.join(data['matched_patterns']),
                'avg_repo_stars': data.get('avg_repo_stars', 0)
            })
        
        pd.DataFrame(td_labels_data).to_csv(csv_file, index=False)
        
        # Print summary
        self._print_summary(report)
        
        logger.info(f"Results saved to {self.output_file} and {csv_file}")

    def _print_summary(self, report: Dict):
        """Print summary to console"""
        print("\n" + "="*60)
        print("TECHNICAL DEBT LABEL DISCOVERY REPORT")
        print("="*60)
        
        metadata = report['analysis_metadata']
        summary = report['repository_summary']
        td_findings = report['technical_debt_findings']
        
        print(f"Analysis Date: {metadata['timestamp']}")
        print(f"Repositories Analyzed: {metadata['total_repositories_analyzed']}")
        print(f"Unique Labels Found: {metadata['total_unique_labels_found']}")
        print(f"Average Stars per Repo: {summary['avg_stars']:.1f}")
        print(f"Average Labels per Repo: {summary['avg_labels_per_repo']:.1f}")
        
        print(f"\nTECHNICAL DEBT LABELS:")
        print(f"Definite TD Labels: {td_findings['definite_td_labels_count']}")
        print(f"Potential TD Labels: {td_findings['potential_td_labels_count']}")
        print(f"TD Adoption Rate: {td_findings['td_adoption_rate']:.1f}%")
        
        print(f"\nMOST COMMON TD LABELS:")
        for i, (label, data) in enumerate(td_findings['most_common_td_labels'][:5], 1):
            print(f"{i}. '{label}' - used in {data['frequency']} repositories")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Discover technical debt labels in LLM/AI repositories')
    parser.add_argument('--token', required=True, help='GitHub API token')
    parser.add_argument('--output', default='labels_discovery.json', help='Output file')
    parser.add_argument('--max_repos_per_term', type=int, default=50, help='Max repositories per search term')
    
    args = parser.parse_args()
    
    # Initialize discoverer
    discoverer = TechnicalDebtLabelDiscoverer(args.token, args.output)
    
    try:
        # Step 1: Discover LLM/AI repositories
        repositories = discoverer.discover_llm_ai_repositories(args.max_repos_per_term)
        
        if not repositories:
            logger.error("No repositories found. Check your token and search criteria.")
            return
        
        # Step 2: Discover all labels
        all_labels_data = discoverer.discover_all_labels(repositories)
        
        # Step 3: Identify technical debt labels
        td_analysis = discoverer.identify_technical_debt_labels(all_labels_data)
        
        # Step 4: Analyze LLM-specific patterns
        llm_analysis = discoverer.analyze_llm_specific_patterns(all_labels_data)
        
        # Step 5: Generate comprehensive report
        report = discoverer.generate_comprehensive_report(
            repositories, all_labels_data, td_analysis, llm_analysis
        )
        
        # Step 6: Save results
        discoverer.save_results(report)
        
        logger.info("Label discovery completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Discovery interrupted by user")
    except Exception as e:
        logger.error(f"Error during discovery: {e}")
        raise

if __name__ == "__main__":
    main()