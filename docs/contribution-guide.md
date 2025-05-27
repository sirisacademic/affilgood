# Contribution Guide

We welcome contributions to the AffilGood project! This guide provides information on how to contribute effectively to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Branching Model](#branching-model)
4. [Contributing Code](#contributing-code)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Issue Reporting](#issue-reporting)
10. [Community Guidelines](#community-guidelines)

## Getting Started

### Before You Begin

Before contributing to AffilGood, please:

1. Read the [README.md](../README.md) to understand the project
2. Review existing [issues](https://github.com/sirisacademic/affilgood/issues) and [pull requests](https://github.com/sirisacademic/affilgood/pulls)
3. Check if your contribution aligns with the project's goals
4. Consider opening an issue to discuss major changes before implementing them

### Ways to Contribute

You can contribute to AffilGood in several ways:

- **Bug Reports**: Report bugs or issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Implement bug fixes, new features, or improvements
- **Documentation**: Improve or add documentation
- **Testing**: Add or improve test coverage
- **Examples**: Provide usage examples or tutorials
- **Performance**: Optimize existing code for better performance

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA-compatible GPU for model acceleration

### Fork and Clone

1. Fork the AffilGood repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/affilgood.git
cd affilgood
```

3. Add the upstream repository as a remote:

```bash
git remote add upstream https://github.com/sirisacademic/affilgood.git
```

### Environment Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:

```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Install additional development tools
pip install pytest pytest-cov black flake8 mypy pre-commit
```

3. Set up pre-commit hooks (optional but recommended):

```bash
pre-commit install
```

### Verify Installation

Test that your development setup works:

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()

# Test with a simple affiliation
result = affil_good.process(["Stanford University, CA, USA"])
print("Setup successful!" if result else "Setup needs debugging")
```

## Branching Model

We use a two-branch model:

### Main Branches

- **`develop`**: Default branch for development and new features
- **`main`**: Production branch for stable releases

### Feature Branches

Create feature branches from `develop`:

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

### Branch Naming Conventions

Use descriptive branch names with prefixes:

- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical fixes for production
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements

Examples:
- `feature/add-wikidata-support`
- `bugfix/fix-memory-leak-in-dense-linker`
- `docs/update-getting-started-guide`

## Contributing Code

### 1. Choose What to Work On

- Check the [Issues](https://github.com/sirisacademic/affilgood/issues) page for open issues
- Look for issues labeled `good first issue` if you're new to the project
- Issues labeled `help wanted` are particularly welcoming to contributions
- For major features, consider opening an issue first to discuss the approach

### 2. Development Workflow

1. **Update your local repository**:
```bash
git checkout develop
git pull upstream develop
```

2. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes**:
   - Write your code following the [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation if needed

4. **Test your changes**:
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=affilgood

# Run linting
flake8 affilgood/
black --check affilgood/
```

5. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: brief description of changes"
```

6. **Push your branch**:
```bash
git push origin feature/your-feature-name
```

### 3. Keep Your Branch Updated

Regularly sync with upstream to avoid conflicts:

```bash
git checkout develop
git pull upstream develop
git checkout feature/your-feature-name
git rebase develop
```

## Pull Request Process

### 1. Prepare Your Pull Request

Before opening a pull request:

- [ ] Ensure all tests pass
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Follow coding standards
- [ ] Rebase on latest `develop` branch
- [ ] Write a clear pull request description

### 2. Open a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Set the base branch to `develop` (not `main`)
4. Fill out the pull request template

### 3. Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows the project coding standards
- [ ] Self-review of the code completed
- [ ] Documentation updated if needed
- [ ] No breaking changes (or breaking changes are documented)
```

### 4. Review Process

1. **Automated Checks**: CI/CD will run automated tests
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Once approved, maintainers will merge your PR

### 5. After Your PR is Merged

1. Delete your feature branch:
```bash
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

2. Update your local repository:
```bash
git checkout develop
git pull upstream develop
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: Maximum 100 characters (not 79)
- **Imports**: Use absolute imports when possible
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format your code
black affilgood/

# Check formatting
black --check affilgood/
```

### Linting

We use **flake8** for linting:

```bash
flake8 affilgood/
```

Configuration in `setup.cfg`:
```ini
[flake8]
max-line-length = 100
exclude = .git,__pycache__,docs/,build/,dist/
ignore = E203,W503
```

### Type Hints

Use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union

def process_affiliations(
    affiliations: List[str], 
    return_scores: bool = False
) -> List[Dict[str, Union[str, float]]]:
    """Process a list of affiliation strings.
    
    Args:
        affiliations: List of affiliation strings to process
        return_scores: Whether to include confidence scores
        
    Returns:
        List of dictionaries containing processing results
    """
    pass
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose and behavior
    of the function in more detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When processing fails
        
    Example:
        >>> result = example_function("test", 20)
        >>> print(result)
        True
    """
    pass
```

#### Class Documentation

```python
class ExampleClass:
    """Brief description of the class.
    
    Longer description explaining the purpose and usage of the class.
    
    Attributes:
        attribute1: Description of attribute1
        attribute2: Description of attribute2
        
    Example:
        >>> instance = ExampleClass()
        >>> instance.method()
    """
    
    def __init__(self, param1: str):
        """Initialize the class.
        
        Args:
            param1: Description of initialization parameter
        """
        self.attribute1 = param1
```

### Error Handling

Use appropriate exception handling:

```python
# Good: Specific exception handling
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError(f"Operation failed: {e}") from e

# Bad: Catching all exceptions silently
try:
    result = risky_operation()
except:
    pass  # Don't do this
```

### Logging

Use structured logging:

```python
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed information for debugging")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("A serious error occurred")
logger.critical("A very serious error occurred")

# Include context in log messages
logger.info(f"Processing {len(affiliations)} affiliations with model {model_name}")
```

## Testing

### Testing Framework

We use **pytest** for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=affilgood --cov-report=html

# Run specific test file
pytest tests/test_entity_linking.py

# Run specific test
pytest tests/test_entity_linking.py::test_whoosh_linker
```

### Test Structure

Organize tests in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_affilgood.py          # Main class tests
├── test_span_identification.py
├── test_ner.py
├── test_entity_linking.py
├── test_normalization.py
└── data/                      # Test data
    ├── sample_affiliations.txt
    └── expected_results.json
```

### Writing Tests

#### Unit Tests

```python
import pytest
from affilgood.span_identification.simple_span_identifier import SimpleSpanIdentifier

class TestSimpleSpanIdentifier:
    def test_identify_spans_with_semicolon(self):
        """Test span identification with semicolon separator."""
        identifier = SimpleSpanIdentifier(separator=';')
        
        text = "Department of CS; Stanford University; CA, USA"
        result = identifier.identify_spans([text])
        
        expected = {
            "raw_text": text,
            "span_entities": [
                "Department of CS",
                "Stanford University", 
                "CA, USA"
            ]
        }
        
        assert len(result) == 1
        assert result[0]["raw_text"] == expected["raw_text"]
        assert result[0]["span_entities"] == expected["span_entities"]
    
    def test_identify_spans_no_separator(self):
        """Test span identification when separator is not found."""
        identifier = SimpleSpanIdentifier(separator=';')
        
        text = "Stanford University, CA, USA"
        result = identifier.identify_spans([text])
        
        assert len(result) == 1
        assert result[0]["span_entities"] == [text]
```

#### Integration Tests

```python
import pytest
from affilgood import AffilGood

class TestAffilGoodIntegration:
    @pytest.fixture
    def affil_good(self):
        """Create AffilGood instance for testing."""
        return AffilGood(
            span_separator=';',  # Use simple span identification
            entity_linkers='Whoosh',  # Use Whoosh linker
            metadata_normalization=False,  # Disable for faster tests
            verbose=False
        )
    
    def test_end_to_end_processing(self, affil_good):
        """Test complete pipeline processing."""
        affiliations = [
            "Department of Computer Science; Stanford University; CA, USA",
            "Max Planck Institute; Tübingen; Germany"
        ]
        
        results = affil_good.process(affiliations)
        
        assert len(results) == 2
        
        # Check first result
        assert results[0]["raw_text"] == affiliations[0]
        assert len(results[0]["span_entities"]) == 3
        assert "ORG" in results[0]["ner"][0]  # Should find organization
        
        # Check second result
        assert results[1]["raw_text"] == affiliations[1]
        assert len(results[1]["span_entities"]) == 3
```

#### Fixtures

Use fixtures for shared test setup:

```python
# conftest.py
import pytest
from affilgood import AffilGood

@pytest.fixture
def sample_affiliations():
    """Sample affiliation data for testing."""
    return [
        "Department of Computer Science, Stanford University, CA, USA",
        "Max Planck Institute for Intelligent Systems, Tübingen, Germany",
        "University of Oxford, Oxford, UK"
    ]

@pytest.fixture
def affil_good_fast():
    """Fast AffilGood configuration for testing."""
    return AffilGood(
        span_separator=';',
        entity_linkers='Whoosh',
        metadata_normalization=False,
        verbose=False
    )
```

### Test Coverage

Aim for high test coverage:

- **Minimum**: 80% code coverage
- **Target**: 90%+ code coverage
- **Critical paths**: 100% coverage for core functionality

```bash
# Generate coverage report
pytest --cov=affilgood --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Performance Tests

Include performance benchmarks:

```python
import time
import pytest
from affilgood import AffilGood

class TestPerformance:
    def test_processing_speed(self):
        """Test processing speed with medium dataset."""
        affil_good = AffilGood()
        affiliations = ["Stanford University, CA, USA"] * 100
        
        start_time = time.time()
        results = affil_good.process(affiliations)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 100 affiliations in under 30 seconds
        assert processing_time < 30.0
        assert len(results) == 100
```

## Documentation

### Documentation Standards

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include examples where helpful
- Follow the existing documentation structure

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guides**: High-level usage documentation
3. **Technical Documentation**: Architecture and implementation details
4. **Examples**: Practical usage examples

### Building Documentation

If the project uses Sphinx or similar:

```bash
# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Documentation Contributions

When contributing documentation:

1. **Check existing docs**: Ensure you're not duplicating content
2. **Follow the structure**: Use the established documentation organization
3. **Test examples**: Ensure all code examples work
4. **Proofread**: Check for typos and clarity

## Issue Reporting

### Before Reporting

1. **Search existing issues**: Check if the issue has already been reported
2. **Use latest version**: Ensure you're using the latest release
3. **Minimal reproduction**: Create a minimal example that reproduces the issue

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Initialize AffilGood with '...'
2. Process text '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- AffilGood version: [e.g. 0.1.0]
- GPU/CUDA: [if applicable]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions
- **Email**: For private communication with maintainers

### Getting Help

If you need help:

1. **Check documentation**: Review the existing documentation
2. **Search issues**: Look for similar questions in GitHub issues
3. **Ask questions**: Open a new issue with the "question" label

### Recognition

Contributors are recognized in several ways:

- **README.md**: Major contributors are listed
- **Release notes**: Contributions are acknowledged in release notes
- **GitHub**: Contributions are visible in the GitHub contribution graph

### Maintainer Responsibilities

Maintainers will:

- Review pull requests in a timely manner
- Provide constructive feedback
- Help contributors improve their contributions
- Maintain the project's quality standards
- Be responsive to community needs

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow

1. **Feature freeze**: Stop accepting new features for the release
2. **Testing**: Comprehensive testing of the release candidate
3. **Documentation**: Update documentation for new features
4. **Changelog**: Update CHANGELOG.md with release notes
5. **Version bump**: Update version numbers
6. **Tag release**: Create a git tag for the release
7. **Publish**: Publish to PyPI and GitHub releases

### Contributing to Releases

Contributors can help with releases by:

- **Testing**: Test release candidates
- **Documentation**: Review and improve release documentation
- **Bug fixes**: Fix issues found during testing
- **Feedback**: Provide feedback on new features

Thank you for contributing to AffilGood! Your contributions help make this project better for everyone.
