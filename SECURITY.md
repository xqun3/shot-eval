# Security Policy

## Supported versions

Security fixes are applied to the latest released version on the default branch.

## Reporting a vulnerability

Do not open a public issue for a credential leak, prompt-injection finding that can expose private media, or an authorization bypass.

Contact the repository maintainer through the private security contact configured on GitHub. Include:

- a concise description;
- reproduction steps that do not include real secrets or private media;
- affected `shot-eval` version;
- impact assessment.

## Credential policy

This project must never contain:

- API keys, bearer tokens, private keys or service-account JSON;
- cloud project IDs tied to a customer or internal environment;
- internal service endpoints;
- real videos, run outputs, prompt bundles or evaluation reports containing user data.

Use Application Default Credentials, workload identity, a secret manager, or local `.env` files excluded by `.gitignore`.
