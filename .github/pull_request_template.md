name: Pull Request
description: Submit a pull request
title: "[PR] "
body:
  - type: markdown
    attributes:
      value: |
        Thank you for contributing! Please fill in the details below.

  - type: checkboxes
    attributes:
      label: Pre-Submission Checklist
      options:
      - label: I have read the CONTRIBUTING.md guidelines
        required: true
      - label: My code follows the code style of this project
        required: true
      - label: I have tested the changes locally
        required: true
      - label: I have added/updated tests for new functionality
        required: true
      - label: I have updated the documentation
        required: true

  - type: textarea
    attributes:
      label: Description
      description: Describe the changes you've made
      placeholder: What does this PR do?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Related Issue
      description: Link to the issue this PR resolves
      placeholder: "Closes #123"

  - type: textarea
    attributes:
      label: Testing
      description: How have you tested these changes?
      placeholder: Describe the testing approach...
    validations:
      required: true

  - type: textarea
    attributes:
      label: Screenshots / Output
      description: If applicable, add screenshots or console output
      placeholder: Add screenshots or output here

  - type: textarea
    attributes:
      label: Notes
      description: Any additional notes for reviewers
      placeholder: Additional information for reviewers...
