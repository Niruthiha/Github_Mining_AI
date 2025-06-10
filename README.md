# GitHub Technical Debt Mining AI

A comprehensive Python tool for mining and analyzing technical debt in AI/ML repositories on GitHub. This tool is designed for PhD research on technical debt patterns in artificial intelligence and machine learning systems.

## Features

- **Repository Discovery**: Search and collect AI/ML repositories based on specific criteria
- **Technical Debt Analysis**: Identify and analyze various forms of technical debt in codebases
- **Issue Mining**: Extract and analyze GitHub issues related to technical debt
- **Data Export**: Export collected data in multiple formats (CSV, JSON)
- **Rate Limiting**: Intelligent handling of GitHub API rate limits
- **Comprehensive Reporting**: Generate detailed analysis reports

## Requirements

```bash
pip install requests pandas python-dotenv tqdm PyGithub
```

## Setup

1. **GitHub Token**: You'll need a GitHub Personal Access Token with appropriate permissions:
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate a new token with `repo` and `user` scopes
   - Keep your token secure and never commit it to version control

2. **Environment Setup**: Create a `.env` file (not included in repo for security):
   ```
   GITHUB_TOKEN=your_github_token_here
   ```

## Usage

### Basic Usage

```bash
python mining.py --token YOUR_GITHUB_TOKEN --output_dir td_data --max_repos 100
```

### Command Line Options

- `--token`: Your GitHub Personal Access Token
- `--output_dir`: Directory to save collected data (default: td_data)
- `--max_repos`: Maximum number of repositories to analyze (default: 100)

### Example

```bash
python mining.py --token ghp_xxxxxxxxxxxx --output_dir ./results --max_repos 50
```

## Output Files

The tool generates several output files in the specified directory:

- `repositories_YYYYMMDD_HHMMSS.csv/json`: Repository metadata and statistics
- `issues_YYYYMMDD_HHMMSS.csv/json`: Issues data related to technical debt
- `analysis_YYYYMMDD_HHMMSS.json`: Comprehensive analysis results
- `summary_report_YYYYMMDD_HHMMSS.txt`: Human-readable summary report

## Data Collected

### Repository Data
- Repository metadata (name, description, stars, forks, etc.)
- Programming languages used
- Repository size and activity metrics
- Technical debt indicators

### Issues Data
- Issue titles and descriptions
- Labels and assignees
- Creation and closure dates
- Technical debt classification

### Analysis Metrics
- Technical debt patterns
- Common issue types
- Repository health indicators
- Trend analysis

## Research Applications

This tool is particularly useful for:

- **Academic Research**: Studying technical debt patterns in AI/ML systems
- **Industry Analysis**: Understanding common technical debt issues in the field
- **Trend Analysis**: Tracking how technical debt evolves in AI/ML projects
- **Best Practices**: Identifying patterns that lead to better code quality

## Rate Limiting

The tool automatically handles GitHub API rate limits:
- Monitors remaining API calls
- Implements exponential backoff for rate limit exceeded errors
- Provides progress indicators for long-running operations

## Security Notes

- Never commit your GitHub token to version control
- Use environment variables or secure configuration files for tokens
- Ensure your token has minimal required permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{github_td_miner,
  title={GitHub Technical Debt Mining AI},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Github_Mining_AI}
}
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

## Changelog

### v1.0.0
- Initial release
- Basic repository and issue mining functionality
- CSV/JSON export capabilities
- Rate limiting support
