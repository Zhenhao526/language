# Population signaling study

This package extends the tabular positive control from one fixed pair to a
fixed-versus-rotating partner population. It tests whether a single sender's
token remains causally useful to several independent workers and reports
partner-specific code alignment. It uses no LLM, visual model, teacher, or
language prior.

The formal eight-seed matrix is summarized in
[population_formal_结果与下一步.md](population_formal_结果与下一步.md). The
rotating-hidden abundant arm produces a shared code with a large
natural−permuted effect for every worker; rotating-visible is a partner-specific
control, and scarce capacity attenuates the effect.

Run the interface tests from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.population_signaling_study.tests.test_game
```

Freeze and execute a small subset:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.population_signaling_study.runner prepare --out /tmp/population_prepared
PYTHONPATH=. .exp_venv/bin/python -m research_program.population_signaling_study.runner execute --prepared /tmp/population_prepared --out /tmp/population_smoke --updates 100 --seeds 74101 --conditions rotating_hidden_live_abundant
```
