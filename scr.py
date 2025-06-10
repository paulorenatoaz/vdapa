contributing_text = """
# Contributing to vdapa

Welcome! Thank you for your interest in contributing to this project.

## Branches
- Use the **develop** branch for ongoing development.
- Create feature branches named `feature/<name>`.
- Use Pull Requests (PRs) to merge changes into **develop**.
- When ready, create release branches for production.

## Commits
- Write clear, descriptive commit messages.
- Follow conventional commit format if possible.

## Pull Requests
- Ensure your code passes all tests.
- Add tests for new features or bug fixes.
- Review and address comments promptly.

## Reporting issues
- Open GitHub issues with clear steps to reproduce.

Thank you for contributing!
"""

with open("CONTRIBUTING.md", "w") as f:
    f.write(contributing_text.strip() + "\n")

print("CONTRIBUTING.md criado com template básico.")
