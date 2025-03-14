# Contributing to AffilGood

Thank you for your interest in contributing to AffilGood! Following these guidelines helps to communicate that you respect the time of the developers managing and developing this project.

## Table of Contents

- [Branching Model](#branching-model)
- [Feature Branches](#feature-branches)
- [Write a Feature](#write-a-feature)
- [Merge to Staging](#merge-to-staging)
- [Merge to Production](#merge-to-production)
- [Best Practices](#best-practices)
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Tracking](#issue-tracking)

## Branching Model

Instead of a single main branch, we use two branches to record the history of the project:

- `main`: Production branch used to deploy the server components to the production environment.
- `develop`: Development and default branch for new features and bug fixes.

![how-it-works](01%20How%20it%20works.svg)

The production branch `main` stores the official release history, and the `develop` branch serves as an integration branch for new features and bug fixes.

## Feature Branches

Each new feature must reside in its own branch and should be called `feature/{title}`.

![Feature-branches](02%20Feature%20branches.svg)

But, instead of branching off of `main`, feature branches use `develop` as their parent branch.

When a feature is complete, [it gets merged back into `develop`](#merge-to-staging). Features must never interact directly with `main`.

## Write a Feature

1. **Create a new feature branch** based off `develop`.

   ```console
   git checkout develop
   git pull
   git checkout -b feature/{title}
   git push --set-upstream origin feature/{title}
   ```

2. **If other contributors, rebase frequently** to incorporate upstream changes from `develop` branch.

   ```console
   git fetch origin
   git rebase origin/develop
   ```

3. When feature is complete and tests pass, **stage and commit the changes**.

   ```console
   git add --all
   git status
   git commit --verbose
   ```

4. **Write a good commit message**.

5. **Publish changes to your branch**.

   ```console
   git push
   ```

## Merge to Staging

1. **If you've created more than one commit**, squash them into cohesive commits with good messages.
   Then, you would [rebase interactively](https://help.github.com/articles/about-git-rebase/):

   ```console
   git fetch origin
   git rebase -i HEAD~5
   # Rebase the commit messages
   git push --force-with-lease origin feature/{title}
   ```

2. **Merge changes** from your feature branch to `develop`.

   ```console
   git checkout develop
   git merge feature/{title} --ff-only
   git push
   ```

3. **Delete the feature branch** local and remote.

   ```console
   git push origin --delete feature/{title}
   git branch -D feature/{title}
   ```

## Merge to Production

Every time we deploy a new version to the production environment the `main` branch needs to be merged from `develop` branch.

```console
git checkout main
git pull
git merge develop --ff-only
git push
```

## Best Practices

- [All `feature` branches](#feature-branches) are created from `develop`.
- All `feature` branches are named `feature/{title}`.
- Rebase frequently your `feature` branches to incorporate upstream changes from `develop`.
- [Squash multiple trivial commits](https://help.github.com/articles/about-git-rebase/) into a single commit, before merging to `develop`.
- Write a good commit message.
- When a `feature` is complete [it is merged](#merge-to-staging) into the `develop`.
- Avoid merge commits by using a rebase workflow.
- Avoid working on `main` branch except as described in [Merge to Production](#merge-to-production).
- Keep the [CHANGELOG.md](../CHANGELOG.md) file updated anytime you plan to merge changes to any of the main branches.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful and considerate of others.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally
```bash
git clone https://github.com/sirisacademic/affilgood.git
cd affilgood
```

3. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .  # Install in development mode
pip install -r requirements-dev.txt  # Install development dependencies
```

4. Set up pre-commit hooks
```bash
pre-commit install
```

## Coding Standards

We follow PEP 8 coding standards for Python code. Our pre-commit hooks will help enforce these standards.

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Use descriptive variable and function names
- Add docstrings for all functions, classes, and modules
- Include type hints where appropriate

## Testing

We use pytest for testing. All new features should have tests, and all tests should pass before submitting a pull request.

To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=affilgood
```

## Documentation

We use Markdown for documentation. All new features should be documented, and existing documentation should be updated to reflect any changes you've made.

Key documentation files:
- `README.md`: Overview of the project
- `docs/api.md`: API documentation
- `docs/usage.md`: Usage examples
- `docs/technical.md`: Technical details

## Issue Tracking

We use GitHub Issues to track bugs and feature requests. When creating a new issue, please:

1. Check if a similar issue already exists
2. Use a clear and descriptive title
3. Provide a detailed description of the issue
4. Include steps to reproduce (for bugs)
5. Include expected behavior and actual behavior
6. Include version information (Python version, affilgood version, etc.)

## Language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers directly.

Thank you for contributing to AffilGood!
