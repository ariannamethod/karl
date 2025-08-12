# `state.json` Format

The orchestrator stores information about processed files in a JSON file located
next to the module. The document structure is:

```json
{
  "version": 1,
  "files": {
    "/absolute/path": {"hash": "<sha256>", "size": 123}
  }
}
```

* `version` – schema version to allow future migrations.
* `files` – mapping of file paths to their SHA256 hash and byte size.

Legacy files without a `version` key are treated as **version 0** and consist of
just the `files` mapping. The loader automatically migrates these files when
reading.
