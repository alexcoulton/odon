# Application-shell persistence fixtures

These files are compatibility artifacts, not ad hoc test inputs. Keep every readable schema version
represented after adding a migration, and never rewrite an old-version fixture into the newest
shape.

| Fixture | Contract |
| --- | --- |
| `v0-project-missing-extension.json` | The supported v0 shape migrates to canonical v1 without dropping an unavailable extension mount. The UI registry annotates that mount as missing, disconnected, or version-incompatible without mutating the persisted tree. |
| `v1-project.json` | The current canonical project document imports and exports without migration. |
| `v1-single-startup.json` | A current-schema viewer document can be selected as a startup application profile and restores only when single-view mode first becomes active. |
| `v1-corrupt-tree.json` | A structurally corrupt current-schema document is rejected atomically; startup restore installs the protected recovery layout and retains diagnostics. |
| `v99-future.json` | A future incompatible schema returns `UNSUPPORTED` without mutation; startup restore installs protected recovery. |

Version policy:

- schema v0 is read-only compatibility input and normalizes to v1;
- schema v1 is the only emitted and stored canonical form;
- corruption and unsupported versions never partially mutate the active shell;
- missing or incompatible extension contributions remain diagnosable retained mounts; and
- a failed configured startup restore always produces a minimal protected native workspace/canvas.
