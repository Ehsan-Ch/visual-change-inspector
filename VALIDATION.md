# Validation record

Published benchmark: 2026-10-14.

- Local automated checks: **21 passed**, Python 3.12.14 on Windows.
- Benchmark: 16 calibration pairs, 24 held-out pairs, two separate failure controls.
- Held-out pooled changed-pixel IoU: 94.610%; no-edit false alarms: 0/6.
- Both failure controls returned REVIEW with no detection mask.
- Interactive report: checked pair selection, change views, reveal slider, and rejected-pair presentation in a browser.
- Installation and command-line entry point: passed after an editable package installation.
- GitHub publication: 17 project files uploaded and verified.
- GitHub Actions: **blocked before any test step started** in [run 38584321800](https://github.com/Ehsan-Ch/visual-change-inspector/actions/runs/38584321800), for code commit `a6244a10eca9df1eefecd062acb53afbcf2210c2`. GitHub's annotation says: "The job was not started because your account is locked due to a billing issue." The job has zero steps. This is not a passing hosted test run or an observed test failure. The workflow remains available to rerun after the account issue is resolved.

Package versions are recorded in `constraints-tested.txt` and benchmark `metrics.json`. Counts and results describe this controlled synthetic run; they do not measure real defect-inspection performance.
