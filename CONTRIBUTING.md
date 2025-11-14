# Contributing to Nested Learning

Thank you for your interest in contributing to the Nested Learning implementation!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/nested-learning.git
   cd nested-learning
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting: `black src/`
- Use type hints where appropriate
- Add docstrings to all functions and classes

### Testing

- Write tests for new features
- Run tests before submitting PR:
  ```bash
  pytest tests/
  ```

### Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Test results

## Areas for Contribution

- **Optimizers**: Implement additional optimizer variants
- **Memory Systems**: Extend CMS with new update strategies
- **Models**: Complete Titans implementation
- **Experiments**: Add new benchmarks and evaluations
- **Documentation**: Improve docs and examples
- **Performance**: Optimize implementations

## Questions?

Open an issue for questions or discussions!