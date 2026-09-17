# Formal archive: corrected binary object-surface remapping

This archive contains the corrected nine-seed matrix for stable surface-label remapping. It compares capacity-matched `mono4` and staged `dual2`, `joint_history` and `slot_local` reception, identity versus swap mappings, full versus leave-one-goal-out support, and live versus silent channels. The parent sender is frozen when worker 0 is replaced.

`aggregate.json` and `compact_results.json` retain endpoint metrics, learning gains, codebooks, contrasts, and hashes without training trajectories. `parent_shift.json` isolates the incumbent sender's identity-to-swap performance loss. `audit.json` records independent replay of all training logs, checkpoints, and paired streams. The complete merged per-run result is uploaded as `results.json` in the repository sync and is intentionally not retained in the local working tree.

The earlier invalid-control diagnostic is preserved at `../formal_20260918/` with its validity note; it is not combined with these statistics.
