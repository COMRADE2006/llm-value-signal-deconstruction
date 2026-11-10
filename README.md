# Deconstructing Value Survey Signals of Large Language Models

Data, results, and analysis code for the study *Deconstructing Value Survey Signals of
Large Language Models: Consensus Dominance and a Single Semantic Axis*.

The study administers a behavioral situational survey scored under the Schwartz theory of
basic values to 11 large language models in two reasoning modes (22 observations,
51,590 model calls), decomposes the observed score variance into item, observation, and
interaction components, tests whether the presupposed seven-dimension structure is
present in the interaction residuals, and isolates the semantic axis that remains.

Repository contents are limited to what is needed to verify and reproduce the reported
results. Model outputs are released verbatim so that the scoring pipeline can be rerun
from raw responses.

## Contents

| Path | What it is |
| --- | --- |
| `data/final_items_v3.jsonl` | Behavioral situational item bank used in the study |
| `data/schwartz_labels.jsonl` | Item-to-dimension labels and active-item flags |
| `data/final_items_v2.jsonl`, `data/pilot_items.jsonl` | Earlier item-bank versions and pilot set (provenance only) |
| `data/review_*.csv`, `data/rewrite_results_v2.jsonl` | Item screening and rewriting records |
| `data/raw_resp/<model>_think-on.jsonl` | Raw model responses, one file per model and reasoning mode |
| `data/raw_resp/<model>_think-off.jsonl` | Raw model responses, reasoning disabled |
| `data/raw_resp_agree/` | Raw responses from an earlier twelve-dimension administration (superseded) |
| `data/raw_resp_agree_retest/` | Retest responses used for the test–retest analysis |
| `data/response_matrix.csv` | Scored observation-by-item response matrix |
| `data/results/schwartz/` | Derived tables and figures reported in the paper |
| `data/results/agree/` | Derived outputs of the earlier twelve-dimension administration (superseded) |
| `analysis/` | Numbered analysis pipeline |
| `api_caller/` | Provider-specific API clients used for collection |
| `run_models/` | Per-model collection entry points and prompt templates |
| `config.py` | Model list, endpoints, and path configuration |
| `requirements.txt` | Python dependencies |

## Reproducing the analyses

```bash
pip install -r requirements.txt
```

All analyses run against the files already in `data/`; no network access is needed. Only
re-collecting responses requires provider credentials, which are read from environment
variables (see `.env.example`) and are not included in this repository.

The pipeline is ordered by the numeric prefix of each script in `analysis/`. The main
entry points for the results reported in the paper are:

| Script | Analysis |
| --- | --- |
| `step9_retest_reliability.py` | Test–retest agreement across repeated collection |
| `step11_block_strength_ci.py` | Same- versus cross-dimension correlation contrast and its interval |
| `step17_mp7.py` | Marchenko–Pastur spectrum test against a column-shuffled permutation null |
| `step18_recovery_sim_7d.py` | Structure-recovery simulation under planted block strength |
| `step6b_axis_validation.py`, `step6d_axis_replication.py` | Semantic-axis validation and split-half replication |
| `step19_figs_7d.py` | Figures reported in the paper |

Scripts numbered `step1`–`step7` belong to an earlier twelve-dimension analysis that the
seven-dimension pipeline supersedes. They are retained for provenance and are not required
to reproduce the published results.

## Notes on data

Model responses were collected through vendor APIs at fixed decoding parameters; model
versions are recorded per file. Responses are reproduced as returned by the providers,
including any refusals or malformed outputs, because the scoring pipeline handles those
cases explicitly.

The item bank was written and screened for this study; it contains no human-subject data.

## License

Code is released under the MIT License (`LICENSE`). Item bank, response data, and derived
results are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

Citation details will be added on publication.
