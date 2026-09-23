# Third-party data and code

| What | Where | License | Used for |
|---|---|---|---|
| 20 Newsgroups (SetFit mirror) | https://huggingface.co/datasets/SetFit/20_newsgroups | see dataset card | `newsgroups` task |
| deepset/prompt-injections | https://huggingface.co/datasets/deepset/prompt-injections | Apache-2.0 | `injection` task |
| banking77 (mteb parquet mirror) | https://huggingface.co/datasets/mteb/banking77 | CC-BY-4.0 | `banking20` task |
| POPE (lmms-lab mirror) | https://huggingface.co/datasets/lmms-lab-encoder/POPE | MIT for the QA pairs (`RUCAIBox/POPE` LICENSE); images are COCO val2014 under their own terms | `pope` task |
| AI2D (lmms-lab mirror) | https://huggingface.co/datasets/lmms-lab-encoder/ai2d | CC BY-SA for the annotations (AI2, via the AWS Registry of Open Data); diagrams were collected from Google Images and keep their own terms | `ai2d` task |
| Oxford-IIIT Pet (timm mirror) | https://huggingface.co/datasets/timm/oxford-iiit-pet | CC-BY-SA-4.0 (tagged on the mirror, matching the original release) | `pets20` task |
| LocalLLaMA/typed-decisions | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | Apache-2.0 | `bench/tasks/typed_decisions.py`: the Laya / Jev-mode tables, the shipped typed heads, `demo/jev_mode.py`. Its gold is one teacher model's soft label per decision, not a human judgment, so accuracy on it is agreement with that teacher |
| Laya checkpoints (NandhaKishorM/laya) | https://github.com/NandhaKishorM/laya | Apache-2.0 | `bench/providers/laya.py`, run through their own `predict` API on the same decisions |
| NanoJev (TianyuCodings/NanoJev) | https://github.com/TianyuCodings/NanoJev | MIT | `bench/providers/nanojev_maze.py`, `nanojev_native_maze.py`: their frozen maze harness, baseline script and episode data |

Datasets are downloaded at run time, never vendored. `bench/tasks/typed_paraphrases.json` (the rewordings used
by `bench.paraphrase_study` and the demo) was written here and carries the repo license.

Two caveats on the POPE and AI2D rows, recorded rather than assumed. **Neither
HF mirror carries a license tag** (the Pet mirror does), so both claims are traced to upstream (the
`RUCAIBox/POPE` LICENSE file and AI2's own Open Data listing) rather than to
the dataset card a reviewer would glance at. And in both, the license covers
the *annotation* layer more cleanly than every underlying pixel. Downloading at
run time and never vendoring keeps that distinction academic; caching a
processed copy into a released artifact would not, and AI2D's share-alike term
would then bite.

Checked and **rejected on license** while choosing those two, so the next
person does not have to re-derive it: MME (academic-research-only, email-gated,
and the HF mirror bypasses the application), Q-Bench (S-Lab License 1.0,
non-commercial), SEED-Bench and SEED-Bench-2 (CC BY-NC 4.0), ScienceQA (CC
BY-NC-SA 4.0 — note `derek-thomas/ScienceQA` mis-tags itself `cc-by-sa-4.0` on
the Hub), RealWorldQA (CC BY-ND 4.0: no derivatives, and permutation
marginalization is a derivative of the option list — note the
`lmms-lab-encoder` mirror silently drops the ND term), and MMStar (no license
anywhere, and 493 of its 1,500 items are re-hosted from the CC BY-NC sources
above). MMBench (Apache-2.0), A-OKVQA (Apache-2.0), BLINK, CV-Bench and MMMU
are all clean and remain open options.

# Methods implemented

- Contextual calibration: Zhao et al., ICML 2021, arXiv:2102.09690
- Batch calibration: Zhou et al., ICLR 2024, arXiv:2309.17249
- Permutation debiasing: Zheng et al., ICLR 2024, arXiv:2309.03882
- Temperature scaling: Guo et al., ICML 2017, arXiv:1706.04599
- L2 heads: shrunk linear discriminant analysis and dual-form ridge regression, textbook closed forms (`anyjev/heads.py`)
