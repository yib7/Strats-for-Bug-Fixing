# Security policy

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/yib7/Strats-for-Bug-Fixing/security/advisories/new). Please do not open a
public issue for something exploitable.

Include what you ran, what happened, and the commit you were on. Expect a first reply within about
a week. This is a research project maintained by one person, so there is no formal SLA and no bug
bounty.

## Supported versions

The latest release and the `main` branch. Older tags do not get backported fixes.

## Scope

This is a command-line research tool. It runs locally, serves nothing, listens on no port, and has
no authentication surface or user accounts. The documentation site is static and loads no
third-party script.

In scope: anything in `src/pop/` or `scripts/`, the dependency set in `uv.lock`, and the CI
workflows.

Out of scope: the vendored third-party benchmark programs under `benchmarks/` (they are deliberately
buggy by design; that is what the project measures), and vulnerabilities that require the attacker
to already control the machine running the tool.

## The execution harness, and what it does not protect against

`pop execbench` compiles and runs **untrusted, model-generated Java** against JUnit tests. That is
the point of the harness, and it is the sharpest edge in the project. Be clear about what it does:

- Each candidate is compiled with `javac` and run with `java` as a separate process, from an
  argument list. No shell is involved, so no shell injection path exists.
- Output is streamed through a bounded sliding window rather than buffered, so a patch that prints
  in a loop cannot exhaust memory.
- The JVM runs with `-Xmx2g` and a wall-clock timeout per bug.
- Benchmark names are validated as bare directory components, and every compiled source path is
  confined to its benchmark directory.

What it does **not** do: there is no container, no seccomp filter, no user namespace, and no
filesystem or network sandbox. Java code that runs under the harness runs with the privileges of
the invoking user. Run it on predictions you generated yourself, or in a disposable environment.
Treat a third party's predictions file the way you would treat any executable they sent you.

## Secrets

The project needs no credential to run. `WANDB_API_KEY` is optional and only enables experiment
tracking during training; the CPU reproduction path makes no network requests at all. See
`.env.example` and the network and telemetry section of the documentation site.
