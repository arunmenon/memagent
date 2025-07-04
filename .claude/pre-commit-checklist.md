# Pre-Commit Checklist for Claude

## Before EVERY Commit

### Code Quality
- [ ] Run `black .` for formatting
- [ ] Run `mypy .` for type checking
- [ ] Run `pytest` and ensure all tests pass
- [ ] Check for any `TODO` or `FIXME` comments

### Security
- [ ] No API keys in code
- [ ] No hardcoded credentials
- [ ] No sensitive data in logs
- [ ] Environment variables used for secrets

### MemAgent Specific
- [ ] Memory operations use `infer()` method
- [ ] Importance scores validated (1-10)
- [ ] Both sync and async methods implemented
- [ ] Provider follows MemoryProvider protocol
- [ ] Embeddings are cached where appropriate

### Documentation
- [ ] Docstrings for all public methods
- [ ] Type hints on all parameters
- [ ] Complex logic has inline comments
- [ ] README updated if needed

## Commands to Run
```bash
# Format code
black MemAgent/ tests/

# Type check
mypy MemAgent/

# Run tests
pytest -v

# Check for secrets (if installed)
detect-secrets scan --all-files
```