# Third-party data and code

| What | Where | License | Used for |
|---|---|---|---|
| 20 Newsgroups (SetFit mirror) | https://huggingface.co/datasets/SetFit/20_newsgroups | see dataset card | `newsgroups` task |
| deepset/prompt-injections | https://huggingface.co/datasets/deepset/prompt-injections | Apache-2.0 | `injection` task |
| banking77 (mteb parquet mirror) | https://huggingface.co/datasets/mteb/banking77 | CC-BY-4.0 | `banking20` task |

Datasets are downloaded at run time, never vendored.

# Methods implemented

- Contextual calibration: Zhao et al., ICML 2021, arXiv:2102.09690
- Permutation debiasing: Zheng et al., ICLR 2024, arXiv:2309.03882
- Temperature scaling: Guo et al., ICML 2017, arXiv:1706.04599
