# Security Policy

## Supported Versions

Security fixes are provided on a best-effort basis for the latest tagged
release branch and the default branch.

## Reporting a Vulnerability

Please do not open public GitHub issues for security vulnerabilities.

Report vulnerabilities privately to [Abdelaziz Bounhar](https://github.com/BounharAbdelaziz)

Include:
- affected version/commit
- impact and threat model
- proof of concept or reproduction steps
- proposed mitigation if available

We will acknowledge receipt within 5 business days and provide status updates
as triage progresses.

## Disclosure Policy

- We prefer coordinated disclosure.
- Once validated and patched, we will publish an advisory in release notes.
- If a fix requires delayed disclosure, we will communicate expected timing.

## Scope Notes

The project may integrate third-party APIs, model servers, and retrieval backends.
Deployment-specific misconfiguration (credentials exposure, insecure network
setup, permissive access controls) is out of scope for code-level vulnerabilities
but can still be reported for guidance.
