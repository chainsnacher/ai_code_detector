name: Bug Report
description: Report a bug or issue
title: "[BUG] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thank you for reporting a bug! Please fill in as much detail as possible.
  
  - type: checkboxes
    attributes:
      label: Is there an existing issue for this?
      description: Please search to see if an issue already exists for the bug you encountered.
      options:
      - label: I have searched the existing issues
        required: true

  - type: textarea
    attributes:
      label: Current Behavior
      description: A clear and concise description of what the bug is.
      placeholder: Tell us what's happening
    validations:
      required: true

  - type: textarea
    attributes:
      label: Expected Behavior
      description: A clear and concise description of what you expected to happen.
      placeholder: What did you expect to happen?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior.
      placeholder: |
        1. 
        2. 
        3.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Environment
      description: Please provide environment details
      placeholder: |
        - OS: [e.g. Windows 11, Ubuntu 22.04]
        - Python Version: [e.g. 3.10]
        - Package Version: [e.g. 1.0.0]
    validations:
      required: true

  - type: textarea
    attributes:
      label: Screenshots / Error Output
      description: If applicable, add screenshots or error messages
      placeholder: Paste error output here

  - type: textarea
    attributes:
      label: Additional Context
      description: Add any other context about the problem here
      placeholder: Any additional information?
