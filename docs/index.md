---
hide:
  - toc
  - title
---

<style>
/* Home page layout: keep top tabs, hide sidebars/TOC, center content. */
.md-sidebar--primary,
.md-sidebar--secondary,
.md-nav--secondary {
  display: none !important;
}

@media screen and (min-width: 76.25em) {
  .md-main__inner {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center;
  }
}

.md-content {
  margin-left: auto;
  margin-right: auto;
}

.md-content__inner {
  margin-left: auto !important;
  margin-right: auto !important;
  max-width: 56rem;
}

.md-content__inner > h1:first-of-type {
  display: none;
}

.home-logo {
  display: block;
  width: min(520px, 100%);
  margin: 0 0 1.25rem 0;
}
</style>

<img src="assets/logo-wordmark.svg" alt="DialogForge logo" class="home-logo" />

Synthetic multi-turn dialogue generation for grounded datasets, with one CLI for local and distributed runs.

## Key Features
- Grounded user-assistant conversation generation with retrieval and optional web search.
- Configurable online/offline judging, deterministic dedup, and resumable run state.
- Export-ready artifacts for ShareGPT-style datasets and Hugging Face workflows.
- Local mode and distributed mode support (Ray + Postgres + optional managed vLLM).

## Get Started
[Get Started](getting-started.md){ .md-button .md-button--primary }
[Setup / Installation](setup-installation.md){ .md-button }

## Next Steps
- [Configuration](CONFIG_REFERENCE.md)
- [CLI / Usage](CLI_REFERENCE.md)
- [API Reference](reference/index.md)
- [Contributing, Issues, and Pull Requests](contributing.md)
