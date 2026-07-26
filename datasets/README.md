# datasets

Public, real-world datasets used by experiments - as opposed to the
**synthetic** data in [`generators/`](../generators/), which is produced
from a known equation.

Each dataset lives in its own subfolder:

```
datasets/<name>/
    load.py      # downloads (into raw/) and/or loads the data into a clean array/frame
    README.md    # what it is, its source, license, and provenance
    raw/          # downloaded source files - gitignored, fetched on demand
    outputs/      # small derived/cached artifacts that are safe to commit
```

Guidelines:

- **Document the source and license** of every dataset in its `README.md`.
  Only use openly redistributable data.
- **Don't commit large raw files.** `load.py` should download them on demand
  into `raw/` (which is gitignored); commit only small derived artifacts, or
  a manifest/checksum, so a clone stays lightweight.
- **Keep loading deterministic** and side-effect-free where possible, so an
  experiment that depends on a dataset stays reproducible.

_No public datasets have been added yet - this folder documents the
convention for when the first one lands._
