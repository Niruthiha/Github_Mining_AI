#!/usr/bin/env python3
"""
Label-Focused GitHub Issue Extractor
Specifically targets repositories with extensive label usage for technical debt research.
Pre-filters repositories to ensure they have both issues AND labels.
"""

import os
import json
import time
import requests
import argparse
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelFocusedExtractor:
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

        # Repositories KNOWN to have extensive label usage
        # These are large, well-managed projects with formal issue triage
        self.guaranteed_labeled_repos = [
            # AI/ML Libraries with excellent label practices
            'huggingface/transformers',        # Extensive label system
            'langchain-ai/langchain',          # Good issue management
            'microsoft/semantic-kernel',       # Microsoft's labeling standards
            'run-llama/llama_index',          # Well-organized issues
            'gradio-app/gradio',              # Active issue triage
            'streamlit/streamlit',            # Great community management
            'openai/openai-python',           # Official SDK with labels
            'anthropics/anthropic-sdk-python', # Official SDK
            'chroma-core/chroma',             # Vector DB with labels
            'weaviate/weaviate-python-client', # Enterprise-level labeling
            'ollama/ollama',                  # Local LLM tool with good labeling
            
            # Popular ML/AI frameworks known for good labeling
            'pytorch/pytorch',                # Extensive label system
            'tensorflow/tensorflow',          # Google's labeling standards
            'scikit-learn/scikit-learn',      # Mature project with labels
            'ray-project/ray',               # Distributed ML with labels
            'mlflow/mlflow',                 # MLOps with good labeling
            'wandb/wandb',                   # ML tracking with labels
            'optuna/optuna',                 # Hyperparameter optimization
            'dask/dask',                     # Parallel computing
            'prefecthq/prefect',             # Workflow management
            'bentoml/BentoML',               # Model serving
            
            # Computer Vision projects with labels
            'ultralytics/yolov5',            # Popular YOLO implementation
            'ultralytics/ultralytics',       # YOLOv8 and beyond
            'opencv/opencv-python',          # OpenCV Python bindings
            'roboflow/supervision',          # Computer vision tools
            'facebookresearch/detectron2',   # Facebook's CV framework
            
            # NLP projects with good labeling
            'explosion/spaCy',               # Industrial NLP
            'nltk/nltk',                    # Natural Language Toolkit
            'RasaHQ/rasa',                  # Conversational AI
            'huggingface/datasets',         # Dataset library
            'sentence-transformers/sentence-transformers', # Embeddings
            
            # API/Service projects known for labels
            'tiangolo/fastapi',             # Modern API framework
            'pallets/flask',                # Micro web framework
            'encode/httpx',                 # Async HTTP client
            'aio-libs/aiohttp',            # Async web framework
            
            # Jupyter/Data Science with labels
            'jupyter/notebook',             # Jupyter Notebook
            'jupyterlab/jupyterlab',       # JupyterLab
            'pandas-dev/pandas',           # Data manipulation
            'matplotlib/matplotlib',       # Plotting library
            'plotly/plotly.py'             # Interactive plotting
        ]

    def _make_request(self, url: str, params: dict = None) -> dict or None:
        """Request with enhanced error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_duration = max(reset_time - time.time(), 60)
                        logger.warning(f"Rate limit critical. Sleeping {sleep_duration:.0f}s")
                        time.sleep(sleep_duration)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.ConnectionError as e:
                if "Failed to resolve" in str(e):
                    logger.error(f"❌ DNS failed. Check connection.")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 15s... ({attempt + 1}/{max_retries})")
                        time.sleep(15)
                    else:
                        return None
                else:
                    logger.error(f"Connection error: {e}")
                    return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        return None

    def check_repository_label_usage(self, repo_name: str) -> dict:
        """Check if repository has good label usage by sampling recent issues."""
        # First check basic repo info
        repo_data = self._make_request(f"{self.base_url}/repos/{repo_name}")
        if not repo_data:
            return {"suitable": False, "reason": "API_ERROR"}
        
        # Check if issues are enabled and exist
        if not repo_data.get('has_issues', False):
            return {"suitable": False, "reason": "NO_ISSUES_ENABLED"}
        
        issue_count = repo_data.get('open_issues_count', 0)
        if issue_count < 5:
            return {"suitable": False, "reason": "TOO_FEW_ISSUES", "issue_count": issue_count}
        
        # Sample recent issues to check label usage
        issues_url = f"{self.base_url}/repos/{repo_name}/issues"
        params = {
            'state': 'all',
            'per_page': 20,  # Sample 20 recent issues
            'sort': 'updated',
            'direction': 'desc'
        }
        
        recent_issues = self._make_request(issues_url, params)
        if not recent_issues:
            return {"suitable": False, "reason": "CANNOT_FETCH_ISSUES"}
        
        # Filter out pull requests and analyze label usage
        actual_issues = [issue for issue in recent_issues if 'pull_request' not in issue]
        
        if len(actual_issues) < 3:
            return {"suitable": False, "reason": "TOO_FEW_ACTUAL_ISSUES", "issue_count": len(actual_issues)}
        
        # Calculate label statistics
        issues_with_labels = sum(1 for issue in actual_issues if issue.get('labels'))
        total_labels = sum(len(issue.get('labels', [])) for issue in actual_issues)
        label_percentage = (issues_with_labels / len(actual_issues)) * 100
        avg_labels_per_issue = total_labels / len(actual_issues)
        
        # Get repository labels to see if they have an organized label system
        labels_url = f"{self.base_url}/repos/{repo_name}/labels"
        repo_labels = self._make_request(labels_url)
        total_repo_labels = len(repo_labels) if repo_labels else 0
        
        # Determine if repository is suitable
        suitable = (
            label_percentage >= 30 and  # At least 30% of issues have labels
            avg_labels_per_issue >= 0.5 and  # Average 0.5 labels per issue
            total_repo_labels >= 5  # Repository has at least 5 different labels
        )
        
        return {
            "suitable": suitable,
            "reason": "GOOD_LABELS" if suitable else "POOR_LABELING",
            "stats": {
                "total_issues_checked": len(actual_issues),
                "issues_with_labels": issues_with_labels,
                "label_percentage": round(label_percentage, 1),
                "avg_labels_per_issue": round(avg_labels_per_issue, 2),
                "total_repo_labels": total_repo_labels,
                "repo_stars": repo_data.get('stargazers_count', 0),
                "repo_language": repo_data.get('language', ''),
                "repo_description": (repo_data.get('description') or '')[:100]
            }
        }

    def find_well_labeled_repositories(self) -> list:
        """Find repositories with excellent label usage."""
        logger.info("🏷️  Finding repositories with excellent label usage...")
        
        suitable_repos = []
        repo_analysis = []
        
        logger.info(f"📋 Checking {len(self.guaranteed_labeled_repos)} known repositories with good labeling...")
        
        for i, repo_name in enumerate(self.guaranteed_labeled_repos, 1):
            logger.info(f"🔍 [{i}/{len(self.guaranteed_labeled_repos)}] Analyzing {repo_name}...")
            
            analysis = self.check_repository_label_usage(repo_name)
            repo_analysis.append({"repo": repo_name, **analysis})
            
            if analysis["suitable"]:
                suitable_repos.append(repo_name)
                stats = analysis["stats"]
                logger.info(f"✅ {repo_name}: {stats['label_percentage']}% labeled, "
                           f"{stats['total_repo_labels']} label types, "
                           f"{stats['repo_stars']} stars")
            else:
                if "stats" in analysis:
                    stats = analysis["stats"]
                    logger.info(f"❌ {repo_name}: {analysis['reason']} "
                               f"({stats['label_percentage']}% labeled)")
                else:
                    logger.info(f"❌ {repo_name}: {analysis['reason']}")
            
            time.sleep(2)  # Be respectful to API
        
        # Save analysis results
        analysis_file = os.path.join(self.output_dir, 'label_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(repo_analysis, f, indent=2)
        
        # Print summary statistics
        suitable_repos_analysis = [r for r in repo_analysis if r["suitable"]]
        if suitable_repos_analysis:
            avg_label_pct = sum(r["stats"]["label_percentage"] for r in suitable_repos_analysis) / len(suitable_repos_analysis)
            avg_repo_labels = sum(r["stats"]["total_repo_labels"] for r in suitable_repos_analysis) / len(suitable_repos_analysis)
            
            logger.info(f"\n📊 LABEL ANALYSIS SUMMARY:")
            logger.info(f"   • Suitable repositories: {len(suitable_repos)}")
            logger.info(f"   • Average label coverage: {avg_label_pct:.1f}%")
            logger.info(f"   • Average labels per repo: {avg_repo_labels:.1f}")
            logger.info(f"   • Analysis saved to: {analysis_file}")
        
        return suitable_repos

    def extract_labeled_issues(self, repo_name: str, max_issues: int = 200) -> list:
        """Extract issues focusing on those with labels."""
        logger.info(f"🏷️  Extracting labeled issues from {repo_name}...")
        
        issues_data = []
        since_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        params = {
            'state': 'all',
            'since': since_date,
            'per_page': 100,
            'direction': 'desc',
            'sort': 'updated'
        }
        
        url = f"{self.base_url}/repos/{repo_name}/issues"
        page = 1
        
        while len(issues_data) < max_issues:
            params['page'] = page
            issues = self._make_request(url, params)
            
            if not issues:
                break
            
            for issue in tqdm(issues, desc=f"Processing {repo_name} issues (Page {page})"):
                if 'pull_request' in issue:
                    continue
                
                labels = [label.get('name', '') for label in issue.get('labels', [])]
                
                # Get comments (limit to save API calls)
                comments = []
                if issue.get('comments', 0) > 0:
                    comments_data = self._make_request(issue['comments_url'])
                    if comments_data:
                        comments = [{'user': c['user']['login'], 'body': c['body']} 
                                  for c in comments_data[:3]]  # Limit to 3 comments
                
                issues_data.append({
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'author': issue['user']['login'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'labels': labels,
                    'body': issue['body'],
                    'comments': comments
                })
                
                if len(issues_data) >= max_issues:
                    break
            
            if len(issues) < 100:
                break
            
            page += 1
            time.sleep(1)
        
        # Calculate label statistics
        issues_with_labels = [issue for issue in issues_data if issue['labels']]
        total_labels = sum(len(issue['labels']) for issue in issues_data)
        label_coverage = len(issues_with_labels) / len(issues_data) * 100 if issues_data else 0
        
        logger.info(f"✅ Extracted {len(issues_data)} issues from {repo_name}")
        logger.info(f"   • {len(issues_with_labels)} issues have labels ({label_coverage:.1f}% coverage)")
        logger.info(f"   • Total labels: {total_labels}")
        logger.info(f"   • Avg labels per issue: {total_labels/len(issues_data):.2f}")
        
        return issues_data

    def run_label_focused_extraction(self):
        """Run extraction focusing on repositories with good label usage."""
        logger.info("🚀 Starting Label-Focused GitHub Issue Extraction...")
        
        # 1. Find repositories with good label usage
        well_labeled_repos = self.find_well_labeled_repositories()
        
        if not well_labeled_repos:
            logger.error("❌ No repositories with good label usage found!")
            return
        
        logger.info(f"🎯 Found {len(well_labeled_repos)} repositories with excellent label usage!")
        
        # 2. Extract issues from well-labeled repositories
        successful_extractions = 0
        total_issues = 0
        total_labeled_issues = 0
        all_labels = set()
        
        for i, repo_name in enumerate(well_labeled_repos, 1):
            logger.info(f"\n📖 Processing {i}/{len(well_labeled_repos)}: {repo_name}")
            
            # Check if already processed
            safe_filename = repo_name.replace('/', '_') + '_issues.json'
            output_path = os.path.join(self.output_dir, safe_filename)
            
            if os.path.exists(output_path):
                logger.info(f"⏭️  Already processed - skipping")
                continue
            
            # Extract labeled issues
            repo_issues = self.extract_labeled_issues(repo_name)
            
            if repo_issues:
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(repo_issues, f, indent=2, ensure_ascii=False)
                
                # Update statistics
                successful_extractions += 1
                total_issues += len(repo_issues)
                labeled_issues = sum(1 for issue in repo_issues if issue['labels'])
                total_labeled_issues += labeled_issues
                
                # Collect all unique labels
                for issue in repo_issues:
                    all_labels.update(issue['labels'])
                
                logger.info(f"💾 Saved {len(repo_issues)} issues ({labeled_issues} with labels)")
            
            time.sleep(3)  # Pause between repos
        
        # Save label summary
        label_summary = {
            "extraction_date": datetime.now().isoformat(),
            "repositories_processed": successful_extractions,
            "total_issues": total_issues,
            "total_labeled_issues": total_labeled_issues,
            "label_coverage_percentage": round(total_labeled_issues/max(total_issues,1)*100, 2),
            "unique_labels": sorted(list(all_labels)),
            "unique_label_count": len(all_labels)
        }
        
        summary_file = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(label_summary, f, indent=2)
        
        # Final summary
        logger.info(f"\n🎉 LABEL-FOCUSED EXTRACTION COMPLETED!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Repositories processed: {successful_extractions}")
        logger.info(f"   • Total issues extracted: {total_issues}")
        logger.info(f"   • Issues with labels: {total_labeled_issues}")
        logger.info(f"   • Label coverage: {total_labeled_issues/max(total_issues,1)*100:.1f}%")
        logger.info(f"   • Unique labels found: {len(all_labels)}")
        logger.info(f"   • Avg issues per repo: {total_issues/max(successful_extractions,1):.1f}")
        logger.info(f"   • Summary saved to: {summary_file}")
        
        # Show some example labels found
        if all_labels:
            example_labels = sorted(list(all_labels))[:20]
            logger.info(f"📋 Example labels found: {', '.join(example_labels)}")

def main():
    parser = argparse.ArgumentParser(description='Label-Focused GitHub Issue Extractor')
    parser.add_argument('--token', required=True, help='GitHub Personal Access Token')
    parser.add_argument('--output_dir', default='./labeled_issue_data', help='Output directory')
    
    args = parser.parse_args()
    
    extractor = LabelFocusedExtractor(github_token=args.token, output_dir=args.output_dir)
    extractor.run_label_focused_extraction()

if __name__ == "__main__":
    main()