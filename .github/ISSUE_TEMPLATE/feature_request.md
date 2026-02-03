name: Feature Request
description: Suggest a new feature or enhancement
title: "[FEATURE] "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thank you for suggesting an enhancement! Please fill in as much detail as possible.

  - type: checkboxes
    attributes:
      label: Is there an existing feature request for this?
      description: Please search to see if a feature request already exists.
      options:
      - label: I have searched the existing issues
        required: true

  - type: textarea
    attributes:
      label: Description
      description: Provide a clear description of the feature you're requesting
      placeholder: Describe the feature...
    validations:
      required: true

  - type: textarea
    attributes:
      label: Motivation
      description: Why should this feature be added? What problem does it solve?
      placeholder: What's the motivation?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Proposed Solution
      description: How would you like this feature to work?
      placeholder: Describe your proposed solution...
    validations:
      required: true

  - type: textarea
    attributes:
      label: Alternatives
      description: Have you considered alternative solutions?
      placeholder: Describe alternatives...

  - type: textarea
    attributes:
      label: Additional Context
      description: Add any other context or screenshots here
      placeholder: Additional context...
