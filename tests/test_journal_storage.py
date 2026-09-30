"""
Tests for the journaled storage, including crash tests that kill a subprocess
in the middle of the write protocol.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import (
    JournalStorage,
    _JOURNAL_COMMIT,
    _JOURNAL_HEADER,
    _JOURNAL_METADATA_RE,
)

WORKER_PATH = Path(__file__).parent / '_journal_worker.py'

# Import the subprocess worker module so the test can use its constants
spec = importlib.util.spec_from_file_location(
    'tinydb_journal_worker', WORKER_PATH
)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

OLD_JSON = worker.OLD_JSON
NEW_JSON = worker.NEW_JSON
OLD_DATA = worker.OLD_DATA
NEW_DATA = worker.NEW_DATA


def run_worker(path, command, *args):
    return subprocess.run(
        [sys.executable, str(WORKER_PATH), command, str(path), *args],
        capture_output=True,
        text=True,
    )


def reopen_state(path):
    """
    Open the storage in a fresh process (fresh recovery) and return the
    decoded database state.
    """

    result = run_worker(path, 'read')
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_roundtrip(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    assert storage.read() is None

    storage.write(OLD_DATA)
    assert storage.read() == OLD_DATA

    # A completed write leaves no journal or temporary files behind
    assert not os.path.exists(path + '.journal')
    assert not os.path.exists(path + '.journal.tmp')
    assert not os.path.exists(path + '.tmp')

    storage.write(NEW_DATA)
    assert storage.read() == NEW_DATA
    storage.close()


def test_shrinking_payload(tmp_path):
    # The main file must be exactly the new (shorter) payload
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(NEW_DATA)
    storage.write(OLD_DATA)
    storage.close()

    with open(path, 'rb') as handle:
        assert handle.read() == OLD_JSON.encode('utf-8')


def test_create_dirs(tmp_path):
    path = str(tmp_path / 'nested' / 'dir' / 'db.json')
    storage = JournalStorage(path, create_dirs=True)
    storage.write(OLD_DATA)
    storage.close()

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()


def test_durable_without_close(tmp_path):
    # Data must be flushed as part of write(), not only on close()
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(NEW_DATA)
    # Deliberately do not close the storage
    del storage

    assert reopen_state(path) == NEW_DATA


def test_read_only_mode(tmp_path):
    path = str(tmp_path / 'db.json')
    run_worker(path, 'init')

    storage = JournalStorage(path, access_mode='r')
    assert storage.read() == OLD_DATA
    with pytest.raises(IOError):
        storage.write(NEW_DATA)
    storage.close()


def test_kwargs_forwarded_to_json_dumps(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path, sort_keys=True)
    storage.write({'b': 1, 'a': 2})
    storage.close()

    with open(path, 'rb') as handle:
        assert handle.read() == b'{"a": 2, "b": 1}'


def test_tinydb_integration(tmp_path):
    path = str(tmp_path / 'db.json')
    db = TinyDB(path, storage=JournalStorage)
    db.insert_multiple({'int': value} for value in range(5))
    db.close()

    db = TinyDB(path, storage=JournalStorage)
    assert len(db) == 5
    db.update({'int': 42}, where('int') == 0)
    assert db.get(where('int') == 42) == {'int': 42}
    db.close()

    assert reopen_state(path) is not None


def test_with_caching_middleware(tmp_path):
    path = str(tmp_path / 'db.json')
    db = TinyDB(path, storage=CachingMiddleware(JournalStorage))
    db.insert({'name': 'cached'})
    db.close()

    assert reopen_state(path) == {
        '_default': {'1': {'name': 'cached'}}
    }


def write_committed_journal(path, payload):
    """
    Create a complete, checksum-valid journal without installing it, as if the
    process had crashed right after committing the journal.
    """

    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    journal_path = path + '.journal'
    with open(journal_path, 'wb') as handle:
        handle.write(_JOURNAL_HEADER)
        handle.write('{0} {1}\n'.format(digest, len(payload)).encode('ascii'))
        handle.write(payload)
        handle.write(_JOURNAL_COMMIT)


def test_recovery_committed_journal(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    write_committed_journal(path, NEW_JSON.encode('utf-8'))

    # Reopening installs the committed new state
    storage = JournalStorage(path)
    assert storage.read() == NEW_DATA
    storage.close()
    assert not os.path.exists(path + '.journal')


def test_recovery_truncated_journal(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    payload = NEW_JSON.encode('utf-8')
    write_committed_journal(path, payload)

    # Truncate the journal in the middle of the payload
    journal_path = path + '.journal'
    with open(journal_path, 'r+b') as handle:
        handle.truncate(len(_JOURNAL_HEADER) + 70)

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()
    assert not os.path.exists(journal_path)


def test_recovery_truncated_commit_marker(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    payload = NEW_JSON.encode('utf-8')
    write_committed_journal(path, payload)

    journal_path = path + '.journal'
    with open(journal_path, 'r+b') as handle:
        handle.truncate(handle.seek(0, os.SEEK_END) - 2)

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()


def test_recovery_checksum_mismatch(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    payload = NEW_JSON.encode('utf-8')
    write_committed_journal(path, payload)

    # Flip a byte in the payload but keep the journal otherwise complete
    journal_path = path + '.journal'
    with open(journal_path, 'r+b') as handle:
        handle.seek(len(_JOURNAL_HEADER) + 65 + 5)
        handle.write(b'Z')

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()
    assert not os.path.exists(journal_path)

    # The main file itself must still be intact JSON
    with open(path, 'rb') as handle:
        assert json.loads(handle.read().decode('utf-8')) == OLD_DATA


def test_recovery_garbage_journal(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    with open(path + '.journal', 'wb') as handle:
        handle.write(b'this is not a journal at all')

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()


def test_recovery_stale_temp_files(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    with open(path + '.journal.tmp', 'wb') as handle:
        handle.write(b'partial')
    with open(path + '.tmp', 'wb') as handle:
        handle.write(b'partial main')

    storage = JournalStorage(path)
    assert storage.read() == OLD_DATA
    storage.close()
    assert not os.path.exists(path + '.journal.tmp')
    assert not os.path.exists(path + '.tmp')


def test_recovery_idempotent(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalStorage(path)
    storage.write(OLD_DATA)
    storage.close()

    write_committed_journal(path, NEW_JSON.encode('utf-8'))

    first = JournalStorage(path)
    assert first.read() == NEW_DATA
    first.close()

    # Reopen multiple times: the result stays stable and no side files remain
    for _ in range(3):
        again = JournalStorage(path)
        assert again.read() == NEW_DATA
        again.close()

    assert reopen_state(path) == NEW_DATA
    assert not os.path.exists(path + '.journal')


@pytest.mark.parametrize('phase,expect_new', [
    ('staged', False),       # journal temp written, not published
    ('published', False),    # journal complete but no COMMIT record
    ('committed', True),     # complete commit survives
    ('installed', True),     # main file replaced
    ('discarded', True),     # write fully finished
])
def test_subprocess_crash_at_each_stage(tmp_path, phase, expect_new):
    path = tmp_path / 'db.json'

    init = run_worker(path, 'init')
    assert init.returncode == 0, init.stderr

    crashed = run_worker(path, 'crash-write', phase)
    assert crashed.returncode == 7, crashed.stderr

    # Right after the crash the main data file must never be half-written:
    # it contains either exactly the old or exactly the new JSON.
    raw = path.read_bytes()
    assert raw in (OLD_JSON.encode('utf-8'), NEW_JSON.encode('utf-8'))
    assert json.loads(raw.decode('utf-8')) is not None

    # Recovery on reopen yields exactly one well-defined version
    state = reopen_state(path)
    assert state == (NEW_DATA if expect_new else OLD_DATA)

    # Recovery cleaned up all side files
    assert not (tmp_path / 'db.json.journal').exists()
    assert not (tmp_path / 'db.json.journal.tmp').exists()
    assert not (tmp_path / 'db.json.tmp').exists()

    # Repeated recovery has the same deterministic result
    assert reopen_state(path) == state


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='SIGKILL timing test relies on POSIX signals',
)
def test_subprocess_sigkill_during_write(tmp_path):
    path = tmp_path / 'db.json'
    init = run_worker(path, 'init')
    assert init.returncode == 0, init.stderr

    for attempt in range(30):
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_PATH), 'kill-write', str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Kill at varied points of the multi-stage write
        time.sleep(0.01 + (attempt % 12) * 0.05)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        assert proc.returncode == -signal.SIGKILL

        state = reopen_state(path)
        assert state in (OLD_DATA, NEW_DATA)

        raw = path.read_bytes()
        assert raw in (
            OLD_JSON.encode('utf-8'),
            NEW_JSON.encode('utf-8'),
        )

        if state == NEW_DATA:
            # Once a new version is visible it remains stable
            assert reopen_state(path) == NEW_DATA
            return

    pytest.fail('SIGKILL never landed after a complete commit was installed')


def test_metadata_regex_rejects_garbage():
    assert _JOURNAL_METADATA_RE.fullmatch('x' * 64 + ' 12') is None
    assert _JOURNAL_METADATA_RE.fullmatch('a' * 64 + ' 12 extra') is None
