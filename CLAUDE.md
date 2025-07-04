# MemAgent Documentation

## 📚 Documentation Index

### Getting Started
- **[Setup Guide](.claude/setup-guide.md)** - Installation, environment, imports

### Architecture & Design
- **[Architecture Overview](.claude/architecture.md)** - System design and components
- **[API Reference](.claude/api-reference.md)** - Classes, methods, and protocols

### Implementation
- **[Implementation Guide](.claude/implementation-guide.md)** - Building providers and features
- **[Code Standards](.claude/code-standards.md)** - Style guide and conventions

### Development
- **[Testing Guide](.claude/testing-guide.md)** - Writing and running tests
- **[Error Handling](.claude/error-handling.md)** - Common issues and solutions
- **[Performance Guide](.claude/performance.md)** - Optimization patterns

### Project Status
- **[Migration Status](.claude/migration-status.md)** - SQLite → MongoDB progress
- **[Technical Debt](.claude/technical-debt.md)** - Known issues and TODOs

---

## 🚨 Critical Instructions for Claude

### MUST Follow
1. **Always use `infer()` method** - Never call provider methods directly
2. **Importance scores must be 1-10** - Validate all ranges
3. **Use async-first approach** - Prefer `ainfer()` over `infer()`
4. **Follow Protocol patterns** - All providers must implement MemoryProvider
5. **Test everything** - Both sync and async versions need tests

### NEVER Do
1. **Never expose API keys** in code or logs
2. **Never implement mock data** - Use real implementations or TODOs
3. **Never use blocking operations** in async methods
4. **Never skip error handling** - Always handle exceptions properly
5. **Never commit without running tests** - Run pytest first

### Code Review Checklist
Before any commit or PR:
- [ ] No hardcoded credentials
- [ ] All public methods have docstrings
- [ ] Type hints on all functions
- [ ] Both sync/async versions tested
- [ ] Error handling implemented
- [ ] Performance considerations addressed

### When Working on MemAgent
1. Check `.claude/code-standards.md` for style guide
2. Read `.claude/architecture.md` before structural changes
3. Follow `.claude/testing-guide.md` for test patterns
4. Review `.claude/technical-debt.md` for known issues
5. **ALWAYS** review `.claude/danger-zones.md` before writing code
6. Use `.claude/pre-commit-checklist.md` before commits

### Auto-Load for Common Tasks
@.claude/danger-zones.md
@.claude/pre-commit-checklist.md

---

💡 **For Claude**: Load only the documentation relevant to your current task.