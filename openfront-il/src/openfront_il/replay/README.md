# replay/

## Responsibility

Reads the per-tick snapshot files that `engine/bridge/src/snapshotWriter.ts`
wrote to `data/snapshots/{gameId}.jsonl`, and owns the **Python-side
definition** of that on-disk schema.

This package does not generate snapshots — it only reads them. Snapshot
generation is exclusively `engine/bridge`'s job.

## Allowed to import / depend on

- Standard library, `pydantic`, `pyarrow`/`pandas` if streaming benefits
  from it.
- `src/openfront_il/utils/` (paths, logging).

## Must NOT import from

- `features/`, `models/`, `training/`, `evaluation/`.
- `engine/` — this package never touches the engine, only the files it
  already produced.

## The schema-drift problem, and how this package solves it

`engine/bridge/src/types.ts` is the **source of truth** for the snapshot
shape (it's what actually writes the files). `snapshot_schema.py` in this
package is a **hand-maintained mirror** of that shape as `pydantic` models.
Because there's no shared schema language between TS and Python in this
prototype, these two definitions can drift silently.

Mitigation: `tests/unit/replay/test_snapshot_schema.py` (see
`tests/README.md`) loads real fixture snapshot lines (`tests/fixtures/`,
produced by the Node side against a real game) and asserts they validate
against `snapshot_schema.py`. `SnapshotHeader` and `TickSnapshot` both use
`extra="forbid"` (unlike `ingestion/api_contract.py`'s `extra="allow"` for
the external, not-ours-to-control API shape) precisely so this test fails
loudly the moment `types.ts` adds a field this file hasn't mirrored yet —
it is the drift alarm, not a guarantee of byte-for-byte sync at all times.

## Contract

| File | Reads | Provides |
|---|---|---|
| `snapshot_schema.py` | — | `pydantic` models mirroring `types.ts`: `SnapshotHeader`, `TickSnapshot`, `PlayerScalars`, `UnitSnapshot`, `AttackSnapshot`, `AllianceSnapshot`, `LegalityMask`, `Winner` |
| `loader.py` | `data/snapshots/{gameId}.jsonl` | `iter_snapshots(game_id)`, a streaming generator of validated `SnapshotHeader \| TickSnapshot` objects (one line read/validated at a time — real files run hundreds of MB to low GB), and `load_snapshots(game_id)`, a small-fixture/test convenience that loads a whole game into memory |

## Style

Same as `src/openfront_il/README.md`. Keep `snapshot_schema.py` field names
identical to `types.ts` field names (not snake_cased on this side) so a
diff between the two files stays a straightforward visual comparison during
manual sync.
