# Decision rules (set before reading the result)

| Item | Setting | Why |
|---|---|---|
| Primary metric | Conversion = share of users who convert | The page's only job |
| Unit | User (first exposure only) | Avoids double-counting repeat visits |
| Minimum worthwhile effect | **+0.5 pp** absolute (~4% relative) | Smaller gains don't cover the cost of maintaining a second page |
| Significance | α = 0.05, two-sided | A worse page must be detectable too |
| Power | 80% | → **67,673 users per arm** at a 12% baseline |
| Validity check | Sample-ratio mismatch: χ² p < 0.001 ⇒ results invalid | Broken assignment biases every number after it |
| Peeking | Read once, at the planned sample size. Daily charts are for monitoring only | Repeated looks inflate false positives |

## Outcomes
- **Ship**: the 95% CI is above 0 *and* the observed lift ≥ +0.5 pp
- **Iterate**: significant, but below +0.5 pp
- **Do not ship**: the CI's upper bound is below +0.5 pp *and* the test was powered for +0.5 pp, so a worthwhile lift is ruled out
- **Extend**: the CI still contains +0.5 pp, or the test is underpowered
- **Invalid**: SRM check fails
