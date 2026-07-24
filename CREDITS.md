# Credits

This project is MIT licensed (see [LICENSE](LICENSE)). It redistributes third-party material under
`benchmarks/`, and it calls models and datasets it does not redistribute. Everything is listed here
with the license that actually governs it.

## Redistributed in this repository

### QuixBugs (Java subset) — MIT

`benchmarks/quixbugs/` holds 40 buggy programs, 40 reference fixes, 40 JUnit test classes and 2
shared support classes from [jkoppel/QuixBugs](https://github.com/jkoppel/QuixBugs), at commit
`4257f44b0ff1181dedaedee6a447e133219fcebf`.

> Copyright 2017-2019 James Koppel

The full MIT permission notice is vendored verbatim at
[`benchmarks/quixbugs/LICENSE`](benchmarks/quixbugs/LICENSE).

### HumanEval-Java — permissive by lineage, unstated at the snapshot

`benchmarks/humaneval_java/` holds 163 buggy programs, 163 reference fixes and 163 JUnit test
classes from [ASSERT-KTH/human-eval-java](https://github.com/ASSERT-KTH/human-eval-java) at commit
`ed75a3e0e8d0c97a632885d67281b26218a3a57f`. (161 are used; `DO_ALGEBRA` and `STRING_TO_MD5` are
excluded from the manifest for JDK-version reasons, not licensing.)

That snapshot states no license. Both of its upstreams are permissive:

| Layer | Source | License |
|---|---|---|
| Original problems | [openai/human-eval](https://github.com/openai/human-eval) | MIT |
| Java transformation, ASE'23 CLM replication package | [lin-tan/clm](https://github.com/lin-tan/clm) | BSD 3-Clause, Copyright (c) 2023, ASSET research group, Purdue University |
| Snapshot vendored here | ASSERT-KTH/human-eval-java | none stated |

The operative grant for the Java transformation is BSD 3-Clause, which permits redistribution in
source form provided the copyright notice, the condition list and the disclaimer are retained, that
the contributors' names are not used to endorse derived products without permission, and the
warranty disclaimer is reproduced. The full chain is recorded in
[`benchmarks/humaneval_java/PROVENANCE.md`](benchmarks/humaneval_java/PROVENANCE.md).

### JUnit 4.13.2 — Eclipse Public License 1.0

`benchmarks/lib/junit-4.13.2.jar` is the unmodified binary artifact from Maven Central
(`junit:junit:4.13.2`), used by the execution harness to run each bug's tests.

EPL-1.0 section 3 requires that a distributor of object code state the license and tell recipients
how to obtain the source. The license text is at <https://www.eclipse.org/legal/epl-v10.html>, and
the source for this version is at <https://github.com/junit-team/junit4/tree/r4.13.2>. The jar is
byte-identical to the published artifact and nothing in it has been modified.

### Hamcrest Core 1.3 — BSD 3-Clause

`benchmarks/lib/hamcrest-core-1.3.jar` is the unmodified binary artifact from Maven Central
(`org.hamcrest:hamcrest-core:1.3`), a JUnit 4 runtime dependency. Copyright (c) 2000-2006,
www.hamcrest.org. Redistribution in binary form is permitted provided the copyright notice, the
condition list and the disclaimer are reproduced, and the contributors' names are not used for
endorsement without permission. License text: <https://opensource.org/license/bsd-3-clause>.

### Test fixtures derived from the above

`tests/fixtures/smoke_*` are Java methods extracted from the two vendored benchmarks by
`scripts/build_smoke_fixtures.py`, so they inherit those licenses. `tests/fixtures/*_example.csv`
are synthetic numbers written for the tests and do not match any real result.

## Used but not redistributed

No dataset is vendored. The loaders fetch these at runtime, under the source's own terms.

- **CodeXGLUE code-refinement (Java)**, the 52K-pair train split and 6,545-pair test split every
  Track 1 number is computed on. [microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE), MIT.
- **CodeSearchNet (Java)**, the pretraining corpus for arm A.
  [github/CodeSearchNet](https://github.com/github/CodeSearchNet), MIT.

Models, downloaded from Hugging Face at run time:

- **Qwen2.5-Coder-1.5B-Instruct** and **Qwen2.5-Coder-1.5B**, Apache 2.0, used by arms C and D.
- **microsoft/codebert-base**, MIT, the dense retriever for arm C.

Arms A and B use no pretrained checkpoint: `src/pop/models/t5_factory.py` builds a randomly
initialised T5 at t5-small dimensions over a 16,384-token SentencePiece vocabulary trained in this
repo.

## Libraries

The dependency set is declared in `pyproject.toml` and pinned in `uv.lock`. All of it is
permissively licensed: PyTorch, Transformers, Accelerate, Datasets, PEFT, SentencePiece, NumPy,
Pydantic, PyYAML, faiss, bm25s, Weights & Biases, MkDocs, Pygments, Ruff and pytest are BSD, MIT or
Apache 2.0; `codebleu` and `tree-sitter` with `tree-sitter-java` are MIT. `tqdm` is dual MPL-2.0 and
MIT, which imposes no obligation on an unmodified install. There is no GPL, AGPL, LGPL or
share-alike component anywhere in the tree or the lockfile, so the MIT release is unencumbered.

The documentation site is built with MkDocs on the `readthedocs` theme (BSD 3-Clause), with syntax
highlighting rendered at build time by Pygments (BSD 2-Clause). The published site loads no
third-party script, font or stylesheet.
