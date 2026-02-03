# Contribution Guidelines

Thank you for your interest in contributing to the AI Code Detector project!

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Include screenshots if possible**
* **Describe the behavior you observed**
* **Explain which behavior you expected to see instead**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior** and **expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Python styleguides
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use type hints where possible
* Add docstrings to all functions and classes
* Keep functions focused and small

### Documentation Styleguide

* Use Markdown for documentation
* Reference code elements with backticks
* Include code examples where helpful

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai_code_detector.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install development tools: `pip install pytest pytest-cov flake8 black`
7. Create a feature branch: `git checkout -b feature/my-feature`

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/

# Run specific test file
pytest tests/test_basic.py -v
```

## Before Submitting

1. Run tests: `pytest tests/`
2. Format code: `black src/ tests/`
3. Check style: `flake8 src/ tests/`
4. Update documentation
5. Add test cases for new features

## Review Process

1. At least one maintainer review required
2. All tests must pass
3. Code coverage should not decrease
4. Documentation must be updated

## Questions?

Feel free to open an issue with the label "question" or contact the maintainers.

---

**Thank you for contributing! 🎉**
