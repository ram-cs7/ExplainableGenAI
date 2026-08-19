# Contributing to ExplainableGenAI.jl

Thank you for your interest in contributing to `ExplainableGenAI.jl`! We welcome contributions from the community to improve mechanistic interpretability and agentic fault diagnosis tools within the Julia SciML ecosystem.

## How to Contribute

### 1. Reporting Issues
If you encounter bugs, installation issues, or unexpected behavior in the Sparse Autoencoders or Agentic Traceability modules, please open an Issue on GitHub. Include:
- A minimal reproducible example (MWE).
- Your Julia version (`versioninfo()`).
- Package versions (`] st`).

### 2. Submitting Pull Requests
- Fork the repository.
- Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature-name`).
- Make your changes, ensuring code is documented and follows standard Julia style guidelines (e.g., `BlueStyle`).
- Write tests for your new features in the `test/` directory.
- Run the full test suite locally before submitting:
  ```julia
  using Pkg
  Pkg.test()
  ```
- Submit a Pull Request with a clear description of the problem solved or feature added.

## Development Setup
To set up the repository for local development:
```julia
# Clone the repository
git clone <REPOSITORY_URL>
cd ExplainableGenAI

# Start Julia and activate the environment
julia --project=.

# Instantiate dependencies
julia> ]
(ExplainableGenAI) pkg> instantiate
```

## Community Standards
This project follows standard open-source community guidelines. Please ensure all interactions in issues and PRs remain constructive, respectful, and focused on improving the research software.
