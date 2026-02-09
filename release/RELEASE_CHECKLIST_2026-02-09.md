# DialogForge OSS Release Checklist (Week of February 9-13, 2026)

## Monday, February 9, 2026

- [ ] Add legal/community files: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`
- [ ] Link governance docs from README
- [ ] Verify `.gitignore` excludes secrets and runtime artifacts
- [ ] Replace placeholder contact fields in `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `.github/ISSUE_TEMPLATE/config.yml`

Gate:
- [ ] Legal/community docs visible and complete

## Tuesday, February 10, 2026

- [ ] Generate sanitized sample dataset bundle using `scripts/release/prepare_sample_dataset.py`
- [ ] Validate dataset card, stats, and plots
- [ ] Publish sample dataset to Hugging Face

Gate:
- [ ] HF repo contains sample JSONL + dataset card + stats + plots

## Wednesday, February 11, 2026

- [ ] Build package artifacts
- [ ] Publish `v0.1.0` to PyPI
- [ ] Create Git tag and GitHub release

Gate:
- [ ] `pip install dlgforge` (or fallback name) and `dlgforge --help` succeed

## Thursday, February 12, 2026

- [ ] Final docs polish in `README.md` and `INSTALL.md`
- [ ] Confirm support boundaries and known limitations are explicit
- [ ] Finalize launch links in release post templates

Gate:
- [ ] Clean-room docs flow works without hidden setup

## Friday, February 13, 2026

- [ ] Set repository visibility to public
- [ ] Publish GitHub release post and LinkedIn post
- [ ] Open and pin onboarding issue
- [ ] Ensure issue templates are enabled

Gate:
- [ ] All launch links live and triage workflow active
