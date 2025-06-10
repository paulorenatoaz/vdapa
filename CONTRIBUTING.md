# Contributing to vdapa

## English Version

Thank you for your interest in contributing to the vdapa project!

### Branches

- Use the **develop** branch for ongoing development.
- Create feature branches off **develop** named `feature/<feature-name>`.
- Use Pull Requests (PRs) to merge your changes into **develop**.
- Prepare release branches from **develop** when ready for production.

### Commits and Pull Requests

- Write clear and descriptive commit messages.
- Ensure your code passes all tests.
- Add tests for new features or bug fixes.
- Review comments and update your PR accordingly.

### Reporting Issues

- Please open issues with detailed steps to reproduce any bugs.

### How to contribute (basic commands)

```bash
# Clone the repository
git clone https://github.com/paulorenatoaz/vdapa.git

# Enter project directory
cd vdapa

# Create and switch to a new feature branch
git checkout -b feature/my-feature develop

# Add your changes
git add .

# Commit with a meaningful message
git commit -m "Describe your changes here"

# Push branch to remote
git push origin feature/my-feature

# Open a Pull Request on GitHub to merge your feature into develop
