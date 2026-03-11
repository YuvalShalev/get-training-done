# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email [yuval@shalev.dev](mailto:yuval@shalev.dev) with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact

You should receive an initial response within 48 hours.

## Scope

Get Training Done is a Claude Code plugin that runs **entirely on your local machine**. Key security properties:

- **No user data is sent externally** — all training happens locally
- **Read-only external API calls** — research tools (arXiv, Kaggle, Papers with Code) only fetch public metadata
- **No authentication tokens stored** — plugin uses the host Claude Code session
- **No network listeners** — MCP servers communicate via stdio, not HTTP

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |
