# Proposed manuscript and reviewer wording

Use the following wording only after a public repository URL exists and the listed contents are actually uploaded. Replace both placeholders with verified values. Do not claim adapters, poisoned datasets, exact historical environments or revised results are public when they are absent.

## Code Availability / Data Availability

The implementation scripts, experimental configurations, and revised evaluation protocol are publicly available at [REPOSITORY_URL] (release/commit: [VERIFIED_REFERENCE]). The repository distinguishes the historical experimental pipeline from the revised evaluation and documents its assumptions and limitations. The pretrained model and source datasets are obtained from their respective providers, subject to their access requirements and terms. Trained adapters and locally generated experimental artifacts are not included in the initial code release.

## Response to reviewer comment 7

We have made the implementation scripts and experimental configurations publicly available and added a repository reference to the manuscript. The repository provides instructions for preparing the evaluation data, running inference, computing the reported metrics, and identifying the versions and artifacts needed for reproduction. Historical and revised evaluation procedures are explicitly distinguished.

Only retain the final sentence about metric reproduction when the completed result files and corresponding scripts have been checked against the revised manuscript.
