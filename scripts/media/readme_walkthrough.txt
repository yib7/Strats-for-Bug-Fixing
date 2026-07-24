# Storyboard for docs/media/reproduce.gif, rendered by scripts/media/make_readme_gif.py.
#
# Every output line below is real stdout/stderr, captured by running the command above it in a
# clean checkout on Windows 11 (Python 3.12, Temurin 21). Two edits were applied, and only these
# two:
#
#   1. The clone's absolute path is rewritten to C:\code\Strats-for-Bug-Fixing, so the recording
#      carries no home directory.
#   2. Long runs of progress output are cut, and every cut is marked in the recording by a `~`
#      line stating how much was removed. No output line is reworded, reordered or invented.
#
# Directives:
#   $ cmd    type this command at the prompt
#   > text   output line
#   - text   output line, dimmed (progress/log noise)
#   * text   output line, highlighted (the result worth reading)
#   ~ text   annotation from this storyboard, not program output
#   . ms     hold the screen for ms milliseconds

# ---------------------------------------------------------------------------------------------
# Beat 1: the whole training pipeline, end to end, on committed fixtures.
# ---------------------------------------------------------------------------------------------
$ uv run pop smoke
- WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
- I0000 00:00:1784934477.867279   14184 sentencepiece_trainer.cc:105] Starts training with :
- trainer_spec {
-   input_format:
-   model_prefix: outputs\smoke\tokenizer
-   model_type: UNIGRAM
-   vocab_size: 512
~ [ 297 of 318 lines cut here: sentencepiece EM training, transformers, tqdm ]
- Writing model shards: 100%|██████████| 1/1 [00:00<00:00, 493.33it/s]
- Loading weights: 100%|██████████| 47/47 [00:00<00:00, 8530.54it/s]
>
* pop smoke -- summary
> ----------------------------------------
> corpus methods               200
> finetune pairs                50
> eval samples                  20
> em                        0.0000
> em_raw                    0.0000
> codebleu                  0.2500
> syntax_valid_rate         0.0000
> ----------------------------------------
* results: C:\code\Strats-for-Bug-Fixing\results\smoke_local.json
. 2800

# ---------------------------------------------------------------------------------------------
# Beat 2: re-render every figure in the report from the committed measurements.
# ---------------------------------------------------------------------------------------------
$ uv run python scripts/figures/make_all.py
* wrote C:\code\Strats-for-Bug-Fixing\docs\figures\four_arm_comparison.png
* wrote C:\code\Strats-for-Bug-Fixing\docs\figures\scaling_curves.png
* wrote C:\code\Strats-for-Bug-Fixing\docs\figures\execution_vs_codebleu.png
. 1700

# ---------------------------------------------------------------------------------------------
# Beat 3: build the study site.
# ---------------------------------------------------------------------------------------------
$ uv run python -m mkdocs build
- INFO    -  Cleaning site directory
- INFO    -  Building documentation to directory: C:\code\Strats-for-Bug-Fixing\site
* INFO    -  Documentation built in 0.16 seconds
. 1700

# ---------------------------------------------------------------------------------------------
# Beat 4: the test suite.
# ---------------------------------------------------------------------------------------------
$ uv run pytest -q
> ............................................................................................ [ 23%]
> ............................................................................................ [ 47%]
> ............................................................................................ [ 71%]
> ............................................................................................ [ 95%]
> ................                                                                             [100%]
~ [ warnings summary cut: 11 torch pin_memory UserWarnings ]
* 384 passed, 11 warnings in 55.51s
. 2400

# ---------------------------------------------------------------------------------------------
# Beat 5: the Java execution harness, checked against the benchmarks' own reference patches.
# ---------------------------------------------------------------------------------------------
$ uv run pop execbench --validate-references --jobs 4
> Wrote results to C:\code\Strats-for-Bug-Fixing\results\execbench_local_validate_references.json
> {
*   "n": 201,
*   "compile_rate": 1.0,
*   "pass_rate": 1.0,
>   "per_benchmark": {
>     "quixbugs": {
>       "n": 40,
>       "compile_rate": 1.0,
>       "pass_rate": 1.0,
>       "error_kind_counts": {
>         "ok": 40
>       }
>     },
>     "humaneval_java": {
>       "n": 161,
>       "compile_rate": 1.0,
>       "pass_rate": 1.0,
>       "error_kind_counts": {
>         "ok": 161
>       }
>     }
>   }
> }
. 3400
