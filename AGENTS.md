# Repository instructions

## GitHub Markdown

When creating or editing GitHub issues, pull requests, release notes, or comments, write actual Markdown line breaks. Never submit literal `\\n` escape sequences as visible text. Prefer `gh --body-file` for multi-line content; if a command argument is necessary, use shell syntax that expands to real newlines. After writing a substantive issue or pull-request body, read it back with `gh issue view` or `gh pr view` and verify its formatting.
