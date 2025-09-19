#!/usr/bin/env python3
# coding=utf-8
"""
Enhanced PMD Analyzer - Consolidated Static Code Analysis Tool

This module provides comprehensive static code analysis capabilities,
consolidating functionality that would typically be spread across multiple
analyzer files. It includes pattern detection, code quality metrics,
security vulnerability scanning, and reporting features.
"""

import os
import re
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue found during analysis."""
    rule_name: str
    severity: str
    file_path: str
    line_number: int
    column: int = 0
    message: str = ""
    description: str = ""
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary for serialization."""
        return {
            'rule': self.rule_name,
            'severity': self.severity,
            'file': self.file_path,
            'line': self.line_number,
            'column': self.column,
            'message': self.message,
            'description': self.description,
            'category': self.category
        }


@dataclass
class AnalysisReport:
    """Contains the complete analysis results."""
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    file_count: int = 0
    total_lines: int = 0
    execution_time: float = 0.0
    
    def add_issue(self, issue: CodeIssue) -> None:
        """Add a new issue to the report."""
        self.issues.append(issue)
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        return Counter(issue.severity for issue in self.issues)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get count of issues by category."""
        return Counter(issue.category for issue in self.issues)


class BaseAnalyzer:
    """Base class for all code analyzers."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.issues: List[CodeIssue] = []
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze a single file and return found issues."""
        raise NotImplementedError("Subclasses must implement analyze_file method")
    
    def is_applicable(self, file_path: str) -> bool:
        """Check if this analyzer is applicable to the given file."""
        return True


class PatternAnalyzer(BaseAnalyzer):
    """Analyzes code for problematic patterns using regex."""
    
    def __init__(self):
        super().__init__("PatternAnalyzer")
        self.patterns = {
            'hardcoded_password': {
                'pattern': r'password\s*=\s*["\'][^"\']+["\']',
                'severity': 'HIGH',
                'message': 'Hardcoded password detected',
                'category': 'security'
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*["\'].*%.*["\']',
                'severity': 'HIGH',
                'message': 'Potential SQL injection vulnerability',
                'category': 'security'
            },
            'unused_import': {
                'pattern': r'^import\s+(\w+)(?:\s+as\s+\w+)?$',
                'severity': 'MEDIUM',
                'message': 'Potentially unused import',
                'category': 'maintainability'
            },
            'long_line': {
                'pattern': r'^.{121,}$',
                'severity': 'LOW',
                'message': 'Line exceeds 120 characters',
                'category': 'style'
            },
            'magic_number': {
                'pattern': r'\b(?<![\w\.])\d{2,}\b(?![\w\.])',
                'severity': 'MEDIUM',
                'message': 'Magic number detected',
                'category': 'maintainability'
            }
        }
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze file content for pattern violations."""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for rule_name, rule_config in self.patterns.items():
                if re.search(rule_config['pattern'], line, re.IGNORECASE):
                    issue = CodeIssue(
                        rule_name=rule_name,
                        severity=rule_config['severity'],
                        file_path=file_path,
                        line_number=line_num,
                        message=rule_config['message'],
                        category=rule_config['category']
                    )
                    issues.append(issue)
        
        return issues
    
    def is_applicable(self, file_path: str) -> bool:
        """Apply to Python files."""
        return file_path.endswith('.py')


class SecurityAnalyzer(BaseAnalyzer):
    """Specialized analyzer for security vulnerabilities."""
    
    def __init__(self):
        super().__init__("SecurityAnalyzer")
        self.security_patterns = {
            'eval_usage': r'\beval\s*\(',
            'exec_usage': r'\bexec\s*\(',
            'pickle_load': r'pickle\.load\s*\(',
            'shell_injection': r'os\.system\s*\(',
            'path_traversal': r'\.\./.*',
            'weak_crypto': r'md5|sha1(?!224|256|384|512)',
        }
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze for security vulnerabilities."""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for vuln_name, pattern in self.security_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issue = CodeIssue(
                        rule_name=f"security_{vuln_name}",
                        severity='HIGH',
                        file_path=file_path,
                        line_number=line_num,
                        message=f"Security vulnerability: {vuln_name}",
                        category='security'
                    )
                    issues.append(issue)
        
        return issues


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzes code complexity metrics."""
    
    def __init__(self):
        super().__init__("ComplexityAnalyzer")
        self.max_function_lines = 50
        self.max_class_lines = 200
        self.max_cyclomatic_complexity = 10
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze complexity metrics."""
        issues = []
        lines = content.split('\n')
        
        # Analyze function length
        function_starts = []
        class_starts = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('def '):
                function_starts.append((line_num, 'function'))
            elif stripped.startswith('class '):
                class_starts.append((line_num, 'class'))
        
        # Check function lengths
        for i, (start_line, block_type) in enumerate(function_starts):
            end_line = len(lines)
            if i + 1 < len(function_starts):
                end_line = function_starts[i + 1][0] - 1
            
            length = end_line - start_line + 1
            max_length = self.max_function_lines if block_type == 'function' else self.max_class_lines
            
            if length > max_length:
                issue = CodeIssue(
                    rule_name=f"long_{block_type}",
                    severity='MEDIUM',
                    file_path=file_path,
                    line_number=start_line,
                    message=f"{block_type.capitalize()} is too long ({length} lines)",
                    category='complexity'
                )
                issues.append(issue)
        
        return issues


class DuplicationAnalyzer(BaseAnalyzer):
    """Detects code duplication."""
    
    def __init__(self):
        super().__init__("DuplicationAnalyzer")
        self.min_duplicate_lines = 5
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze for code duplication within a file."""
        issues = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Simple duplication detection
        seen_blocks = defaultdict(list)
        
        for i in range(len(lines) - self.min_duplicate_lines + 1):
            block = tuple(lines[i:i + self.min_duplicate_lines])
            seen_blocks[block].append(i + 1)
        
        for block, occurrences in seen_blocks.items():
            if len(occurrences) > 1:
                for line_num in occurrences:
                    issue = CodeIssue(
                        rule_name="code_duplication",
                        severity='MEDIUM',
                        file_path=file_path,
                        line_number=line_num,
                        message=f"Duplicate code block detected",
                        category='duplication'
                    )
                    issues.append(issue)
        
        return issues


class MetricsCalculator:
    """Calculates various code metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(int)
    
    def calculate_file_metrics(self, file_path: str, content: str) -> Dict[str, Any]:
        """Calculate metrics for a single file."""
        lines = content.split('\n')
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'function_count': len(re.findall(r'^\s*def\s+', content, re.MULTILINE)),
            'class_count': len(re.findall(r'^\s*class\s+', content, re.MULTILINE)),
            'import_count': len(re.findall(r'^\s*(?:import|from)\s+', content, re.MULTILINE)),
        }
        
        return metrics


class ReportGenerator:
    """Generates analysis reports in various formats."""
    
    def __init__(self):
        pass
    
    def generate_console_report(self, report: AnalysisReport) -> str:
        """Generate a console-friendly report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Enhanced PMD Analysis Report")
        lines.append("=" * 60)
        lines.append(f"Files analyzed: {report.file_count}")
        lines.append(f"Total lines: {report.total_lines}")
        lines.append(f"Execution time: {report.execution_time:.2f}s")
        lines.append("")
        
        severity_counts = report.get_severity_counts()
        lines.append("Issues by severity:")
        for severity, count in severity_counts.items():
            lines.append(f"  {severity}: {count}")
        lines.append("")
        
        category_counts = report.get_category_counts()
        lines.append("Issues by category:")
        for category, count in category_counts.items():
            lines.append(f"  {category}: {count}")
        lines.append("")
        
        if report.issues:
            lines.append("Issues found:")
            lines.append("-" * 40)
            for issue in sorted(report.issues, key=lambda x: (x.file_path, x.line_number)):
                lines.append(f"{issue.file_path}:{issue.line_number} [{issue.severity}] {issue.message}")
        else:
            lines.append("No issues found!")
        
        return "\n".join(lines)
    
    def generate_json_report(self, report: AnalysisReport) -> str:
        """Generate a JSON report."""
        data = {
            'summary': {
                'file_count': report.file_count,
                'total_lines': report.total_lines,
                'execution_time': report.execution_time,
                'issue_count': len(report.issues),
                'severity_counts': report.get_severity_counts(),
                'category_counts': report.get_category_counts()
            },
            'metrics': report.metrics,
            'issues': [issue.to_dict() for issue in report.issues]
        }
        return json.dumps(data, indent=2)


class EnhancedPMDAnalyzer:
    """Main analyzer class that coordinates all analysis activities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analyzers: List[BaseAnalyzer] = []
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()
        
        # Initialize analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self) -> None:
        """Initialize all analyzers."""
        self.analyzers = [
            PatternAnalyzer(),
            SecurityAnalyzer(),
            ComplexityAnalyzer(),
            DuplicationAnalyzer(),
        ]
        
        logger.info(f"Initialized {len(self.analyzers)} analyzers")
    
    def analyze_file(self, file_path: str) -> List[CodeIssue]:
        """Analyze a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        all_issues = []
        for analyzer in self.analyzers:
            if analyzer.enabled and analyzer.is_applicable(file_path):
                try:
                    issues = analyzer.analyze_file(file_path, content)
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Error in {analyzer.name} analyzing {file_path}: {e}")
        
        return all_issues
    
    def analyze_directory(self, directory: str, patterns: List[str] = None) -> AnalysisReport:
        """Analyze all files in a directory."""
        if patterns is None:
            patterns = ['*.py']
        
        report = AnalysisReport()
        start_time = __import__('time').time()
        
        directory_path = Path(directory)
        files_to_analyze = []
        
        for pattern in patterns:
            files_to_analyze.extend(directory_path.rglob(pattern))
        
        logger.info(f"Found {len(files_to_analyze)} files to analyze")
        
        total_lines = 0
        for file_path in files_to_analyze:
            if file_path.is_file():
                try:
                    # Analyze file
                    issues = self.analyze_file(str(file_path))
                    report.issues.extend(issues)
                    
                    # Calculate metrics
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_metrics = self.metrics_calculator.calculate_file_metrics(str(file_path), content)
                    total_lines += file_metrics['total_lines']
                    
                    # Add to report metrics
                    for key, value in file_metrics.items():
                        if key not in report.metrics:
                            report.metrics[key] = 0
                        report.metrics[key] += value
                    
                    report.file_count += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")
        
        report.total_lines = total_lines
        report.execution_time = __import__('time').time() - start_time
        
        logger.info(f"Analysis complete: {len(report.issues)} issues found in {report.file_count} files")
        
        return report
    
    def run_analysis(self, target: str, output_format: str = 'console', output_file: Optional[str] = None) -> None:
        """Run complete analysis and generate report."""
        if os.path.isfile(target):
            # Analyze single file
            issues = self.analyze_file(target)
            report = AnalysisReport(issues=issues, file_count=1)
            
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            report.total_lines = len(content.split('\n'))
            
        elif os.path.isdir(target):
            # Analyze directory
            report = self.analyze_directory(target)
        else:
            logger.error(f"Invalid target: {target}")
            return
        
        # Generate report
        if output_format == 'json':
            report_content = self.report_generator.generate_json_report(report)
        else:
            report_content = self.report_generator.generate_console_report(report)
        
        # Output report
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write report to {output_file}: {e}")
        else:
            print(report_content)


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        'analyzers': {
            'pattern': {'enabled': True},
            'security': {'enabled': True},
            'complexity': {'enabled': True, 'max_function_lines': 50},
            'duplication': {'enabled': True, 'min_duplicate_lines': 5}
        },
        'file_patterns': ['*.py'],
        'exclude_patterns': ['*test*', '*__pycache__*'],
        'severity_threshold': 'LOW',
        'output': {
            'format': 'console',
            'file': None
        }
    }


def main():
    """Main entry point for the enhanced PMD analyzer."""
    parser = argparse.ArgumentParser(description='Enhanced PMD Analyzer - Comprehensive Static Code Analysis')
    parser.add_argument('target', help='File or directory to analyze')
    parser.add_argument('--format', choices=['console', 'json'], default='console',
                       help='Output format (default: console)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--patterns', nargs='+', default=['*.py'],
                       help='File patterns to analyze (default: *.py)')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file {args.config}: {e}")
            config = create_sample_config()
    else:
        config = create_sample_config()
    
    # Validate target
    if not os.path.exists(args.target):
        logger.error(f"Target path does not exist: {args.target}")
        sys.exit(1)
    
    # Initialize and run analyzer
    try:
        analyzer = EnhancedPMDAnalyzer(config)
        analyzer.run_analysis(
            target=args.target,
            output_format=args.format,
            output_file=args.output
        )
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()