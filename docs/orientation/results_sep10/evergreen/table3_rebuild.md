| model | Aug r(PPL) | Aug pooled | Aug within-form | rebuilt r(PPL) | rebuilt pooled | rebuilt within-form | rebuilt source |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct | +0.147 | 0.5857 | 0.5806 | +0.147 | 0.5857 | 0.5806 | eg_ao_qwen3_4b_it.jsonl |
| Qwen2.5-7B | +0.179 | 0.6083 | 0.5621 | +0.179 | 0.6083 | 0.5621 | eg_qwen25_7b.jsonl |
| gpt-oss-20B | +0.088 | 0.5829 | 0.5552 | -0.175 | 0.3964 | 0.4147 | eg_ao_gptoss_20b.jsonl |
| Qwen3.5-27B | +0.147 | 0.6109 | 0.5833 | +0.181 | 0.6083 | 0.5847 | eg_ao_qwen35_27b.jsonl |
| Qwen3.6-27B | +0.040 | 0.5594 | 0.5222 | +0.114 | 0.5711 | 0.5471 | eg_ao_qwen36_27b.jsonl |
| Qwen3.8-27B | — | — | — | +0.223 | 0.6295 | 0.5986 | eg_ao_qwen38_27b.jsonl |
| Qwen3.6-35B-A3B | +0.069 | 0.5765 | 0.5532 | +0.190 | 0.5839 | 0.5658 | eg_ao_qwen36_35b.jsonl |

| summary | Aug (Sep 2) | rebuilt, same six | rebuilt, without gpt-oss | rebuilt, + Qwen3.8 |
|---|---|---|---|---|
| mean_pooled_auroc | 0.5873 | 0.5589 | 0.5915 | 0.5690 |
| mean_within_auroc | 0.5594 | 0.5425 | 0.5681 | 0.5505 |
| share_of_excess_removed_by_form | 0.32 | 0.28 | 0.26 | 0.27 |
| within_small_mean | 0.5714 | 0.5714 | 0.5714 | 0.5714 |
| within_large_mean | 0.5535 | 0.5281 | 0.5659 | 0.5422 |
| size_difference_large_minus_small | -0.0179 | -0.0433 | -0.0055 | -0.0292 |
| vintage_pair_range | 0.0611 | 0.0376 | 0.0376 | 0.0376 |
| r(PPL) range | +0.040 to +0.179 | -0.175 to +0.190 | +0.114 to +0.190 | -0.175 to +0.223 |
