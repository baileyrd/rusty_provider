# Security Policy

## Reporting a vulnerability
Do **not** open a public issue. Report privately via
[GitHub Security Advisories](https://github.com/baileyrd/rusty_provider/security/advisories/new),
or reach baileyrd@gmail.com directly if advisories aren't set up on this repo yet.

Include: what you found, affected version/commit, reproduction steps or PoC, and
impact as you understand it.

## Response expectations
- Acknowledgment: within a few business days
- Triage/severity assessment: to follow, communicated in the advisory thread
- Fix timeline: depends on severity; coordinated disclosure once a fix is available

## Supported versions
| Version | Supported |
| ------- | --------- |
| latest  | ✅        |
| older   | ❌        |

## Dependency scanning
`cargo audit` runs against `Cargo.lock` on every push/PR that touches a
`Cargo.toml`/`Cargo.lock`, plus daily on a schedule (so a newly published
advisory against an already-pinned dependency doesn't go unnoticed
between pushes) — see `.github/workflows/audit.yml`.
