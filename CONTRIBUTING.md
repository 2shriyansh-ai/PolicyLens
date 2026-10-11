# Contributing

1. Create a focused branch.
2. Install with `python -m pip install -e .`.
3. Run `python -m unittest discover -s tests -v`.
4. Run `python -m compileall -q src app.py`.
5. Do not commit API keys, private policies, generated evaluation artifacts or
   résumé metrics that were not reproduced from the held-out set.

Changes to retrieval, chunking or refusal logic should include an evaluation
comparison and a short failure analysis, not only successful examples.

