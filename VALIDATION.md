# Validation record

Published benchmark: 2026-10-14.

- Local automated checks: **21 passed**, Python 3.12.14 on Windows.
- Benchmark: 16 calibration pairs, 24 held-out pairs, two separate failure controls.
- Held-out pooled changed-pixel IoU: 94.610%; no-edit false alarms: 0/6.
- Both failure controls returned REVIEW with no detection mask.
- Interactive report: checked pair selection, change views, reveal slider, and rejected-pair presentation in a browser.
- GitHub Actions: publication/check status will be recorded after the repository upload. Local success does not establish hosted CI success.

Package versions are recorded in `constraints-tested.txt` and benchmark `metrics.json`. Counts and results describe this controlled synthetic run; they do not measure real defect-inspection performance.
