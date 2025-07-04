# Claude Code Project Template

## For New Projects, Create This Structure:

### 1. Root CLAUDE.md
```markdown
# [Project Name] Documentation

## 📚 Documentation Index

### Getting Started
- **[Setup Guide](.claude/setup-guide.md)** - Installation, environment, imports

### Architecture & Design  
- **[Architecture Overview](.claude/architecture.md)** - System design and components
- **[API Reference](.claude/api-reference.md)** - Classes, methods, and protocols

### Implementation
- **[Implementation Guide](.claude/implementation-guide.md)** - Building new features
- **[Code Standards](.claude/code-standards.md)** - Style guide and conventions

### Development
- **[Testing Guide](.claude/testing-guide.md)** - Writing and running tests
- **[Error Handling](.claude/error-handling.md)** - Common issues and solutions
- **[Performance Guide](.claude/performance.md)** - Optimization patterns

### Project Status
- **[Technical Debt](.claude/technical-debt.md)** - Known issues and TODOs

---

## 🚨 Critical Instructions for Claude

### MUST Follow
1. [Project-specific critical rule 1]
2. [Project-specific critical rule 2]
3. [Project-specific critical rule 3]

### NEVER Do
1. [Project-specific anti-pattern 1]
2. [Project-specific anti-pattern 2]
3. [Project-specific anti-pattern 3]

### Code Review Checklist
Before any commit or PR:
- [ ] [Project-specific check 1]
- [ ] [Project-specific check 2]
- [ ] [Project-specific check 3]

### When Working on [Project Name]
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
```

### 2. Create .claude/ Directory with These Files:

#### .claude/architecture.md
- System overview
- Component diagram
- Data flow
- Key design decisions

#### .claude/code-standards.md
- Language-specific style guide
- Naming conventions
- Project patterns
- Code organization

#### .claude/danger-zones.md
- Common mistakes with examples
- ❌ Wrong vs ✅ Correct patterns
- Security considerations
- Performance pitfalls

#### .claude/pre-commit-checklist.md
- Formatting commands
- Test commands
- Linting commands
- Security checks

#### .claude/setup-guide.md
- Prerequisites
- Installation steps
- Environment variables
- Development setup

#### .claude/testing-guide.md
- Test structure
- Mock patterns
- Coverage requirements
- CI/CD integration

### 3. Optional/As-Needed Files:
- .claude/api-reference.md (for libraries/frameworks)
- .claude/migration-status.md (for ongoing migrations)
- .claude/performance.md (for performance-critical apps)
- .claude/error-handling.md (for complex error scenarios)
- .claude/implementation-guide.md (for complex domains)

## Quick Setup Script

```bash
#!/bin/bash
# setup-claude-docs.sh

# Create .claude directory
mkdir -p .claude

# Create main CLAUDE.md
cat > CLAUDE.md << 'EOF'
# Project Documentation

## 📚 Documentation Index
[... template content ...]
EOF

# Create standard documentation files
touch .claude/architecture.md
touch .claude/code-standards.md
touch .claude/danger-zones.md
touch .claude/pre-commit-checklist.md
touch .claude/setup-guide.md
touch .claude/testing-guide.md
touch .claude/technical-debt.md

echo "Claude documentation structure created!"
echo "Remember to fill in project-specific content in each file."
```

## Key Principles:
1. **CLAUDE.md is the hub** - Keep it concise with links
2. **Detailed docs in .claude/** - Organize by concern
3. **Auto-load critical files** - Use @imports
4. **Project-specific > Generic** - Customize for each project
5. **Keep it maintained** - Update as project evolves