Put original data in `raw/` and converted episodes in `processed/`. Fill
`manifest.example.json` for a new dataset. See [the dataset contract](../docs/datasets.md).
Data files are ignored by Git and are not included in a fresh clone.

The prepared server shares LIBERO demonstrations at
`/workspace/upenn/datasets/physical-intelligence/libero` and robustness assets
under `/workspace/upenn/datasets/`. Online simulator evaluation uses task assets
and initial states rather than reading the demonstration dataset. See the
[robustness guide](../docs/robustness-evaluation.md) for the asset protocol.

DICE requires its own verified, normalized replay export. Its default template
uses `datasets/dice/libero_spatial_0.mpk`; the local smoke configuration instead
points to the existing single-demonstration export in the shared dataset folder.
That small diagnostic export is not a complete training corpus. Follow the
[DICE data instructions](../docs/dice.md#demonstration-data-preparation).

