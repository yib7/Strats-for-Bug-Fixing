# Credits

This project is MIT licensed (see [LICENSE](LICENSE)). It redistributes third-party material under
`benchmarks/`, and it calls models and datasets it does not redistribute. Everything is listed here
with the license that actually governs it.

## Redistributed in this repository

### QuixBugs (Java subset): MIT

`benchmarks/quixbugs/` holds 40 buggy programs, 40 reference fixes, 40 JUnit test classes and 3
shared support classes (`Node`, `WeightedEdge`, `QuixFixOracleHelper`) from
[jkoppel/QuixBugs](https://github.com/jkoppel/QuixBugs), at commit
`4257f44b0ff1181dedaedee6a447e133219fcebf`.

> Copyright 2017-2019 James Koppel

The full MIT permission notice is vendored verbatim at
[`benchmarks/quixbugs/LICENSE`](benchmarks/quixbugs/LICENSE).

### HumanEval-Java: permissive by lineage, unstated at the snapshot

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

The operative grant for the Java transformation is BSD 3-Clause. Its condition 1 requires that a
source redistribution retain the copyright notice, the condition list and the disclaimer, so that
notice is reproduced in full below rather than summarised. The chain and the vendoring detail are
recorded in [`benchmarks/humaneval_java/PROVENANCE.md`](benchmarks/humaneval_java/PROVENANCE.md).

**Why this is redistributed here.** The snapshot's silence is an absent statement, not a refusal:
ASSERT-KTH/human-eval-java presents itself in its own README as a repackaging of the CLM
replication package's Java port, not as new authorship, and both layers it repackages grant
redistribution with attribution. A downstream redistributor who satisfies the upstream conditions —
which is what this section does — is therefore acting inside the only grants that were ever made
over this material. The alternative readings were weighed and rejected: dropping the benchmark
would remove 161 of the 201 execution-harness bugs and with them the study's headline
execution-vs-CodeBLEU comparison, and re-deriving the Java port locally would reproduce the same
CLM-derived artifacts under the same BSD terms with no license improvement. Nothing here is
modified, nothing is relicensed, no `LICENSE` file is fabricated for the snapshot, and all three
layers are named. This is a considered decision, not an open question; if ASSERT-KTH later states
terms, this section gets updated to match them.

Reproduced from [`lin-tan/clm/LICENSE`](https://github.com/lin-tan/clm/blob/main/LICENSE):

```text
BSD 3-Clause License

Developed by:
The ASSET research group led by Lin Tan
Purdue University

Copyright (c) 2023
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

OpenAI's HumanEval, the layer underneath, is MIT: permission is granted free of charge to deal in
the software without restriction, provided the copyright notice and permission notice are included
in all copies or substantial portions. Its notice is at
<https://github.com/openai/human-eval/blob/master/LICENSE>.

### JUnit 4.13.2: Eclipse Public License 1.0

`benchmarks/lib/junit-4.13.2.jar` is the unmodified binary artifact from Maven Central
(`junit:junit:4.13.2`), used by the execution harness to run each bug's tests.

EPL-1.0 section 3 requires that a distributor of object code state the license and tell recipients
how to obtain the source. So, stated plainly: **this jar is distributed under the Eclipse Public
License 1.0**, whose text is at <https://www.eclipse.org/legal/epl-v10.html> and is also carried
inside the jar itself at `LICENSE-junit.txt`; **the corresponding source is at
<https://github.com/junit-team/junit4/tree/r4.13.2>** and from Maven Central as
`junit:junit:4.13.2:sources`. The jar is byte-identical to the published artifact — its SHA-1 is
`8ac9e16d933b6fb43bc7f576336b8f4d7eb5ba12`, matching
<https://repo1.maven.org/maven2/junit/junit/4.13.2/junit-4.13.2.jar.sha1> — and nothing in it has
been modified, so EPL-1.0's reciprocity (which reaches modifications of the EPL-covered files) has
nothing to attach to and this repository's own MIT license is unaffected.

### Hamcrest Core 1.3: BSD 3-Clause

`benchmarks/lib/hamcrest-core-1.3.jar` is the unmodified binary artifact from Maven Central
(`org.hamcrest:hamcrest-core:1.3`), a JUnit 4 runtime dependency. Its SHA-1 is
`42a25dc3219429f0e5d060061f71acb49bf010a0`, matching
<https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar.sha1>.

The BSD 3-Clause condition 2 requires a binary redistribution to reproduce the notice, the
conditions and the disclaimer "in the documentation and/or other materials provided with the
distribution" — this file is that documentation, so the notice is reproduced verbatim from
`LICENSE.txt` inside the jar rather than summarised:

```text
BSD License

Copyright (c) 2000-2006, www.hamcrest.org
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the distribution.

Neither the name of Hamcrest nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```

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

- **Qwen2.5-Coder-1.5B-Instruct**, Apache 2.0. Arm C prompts it; arm D LoRA-adapts it. Both arms use
  this same checkpoint.
- **microsoft/codebert-base**, MIT, the dense retriever for arm C.

Arms A and B use no pretrained checkpoint: `src/pop/models/t5_factory.py` builds a randomly
initialised T5 at t5-small dimensions over a 16,384-token SentencePiece vocabulary trained in this
repo.

## Libraries

The dependency set is declared in `pyproject.toml` and pinned in `uv.lock`. All of it is
permissively licensed: PyTorch, Transformers, Accelerate, Datasets, PEFT, SentencePiece, NumPy,
Pydantic, PyYAML, faiss, bm25s, Weights & Biases, MkDocs, Pygments, Ruff and pytest are BSD, MIT or
Apache 2.0; `codebleu` and `tree-sitter` with `tree-sitter-java` are MIT.

**There is no GPL, LGPL, AGPL or SSPL anywhere** — not in the declared dependencies, not in the
transitive closure of the lockfile, not in the vendored assets. The only reciprocal licenses present
anywhere are file-level ones: MPL-2.0, on `tqdm` (dual-licensed with MIT) and on the transitive
dependencies `certifi` and `pathspec`; and EPL-1.0, on the vendored JUnit jar. Both are per-file
copyleft that binds only someone who modifies the covered files and redistributes the result. Every
one of these arrives as an unmodified upstream artifact — installed by `uv` from PyPI, or copied
byte-for-byte from Maven Central — so none of them reaches this repository's own code, and the MIT
release is unencumbered. Checked by reading `License-Expression`, `License` and the `License ::`
classifiers out of the installed metadata of all 124 distributions in `.venv`, not from memory.

The documentation site is built with MkDocs on the `readthedocs` theme (BSD 3-Clause), with syntax
highlighting rendered at build time by Pygments (BSD 2-Clause). The published site loads no
third-party script, font or stylesheet.
