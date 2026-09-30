"""
Tests for :class:`tinydb.storages.JournalingJSONStorage`.

The crash-recovery tests run the storage in real subprocesses that terminate
themselves (without running cleanup code) at well-defined points of a write,
mirroring ``SIGKILL`` / power-loss style interruptions.
"""

import json
import os
import signal
import struct
import subprocess
import sys
import time
import zlib

import pytest

from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JournalingJSONStorage, Storage

FRAME_MAGIC = JournalingJSONStorage._FRAME_MAGIC
COMMIT_MAGIC = JournalingJSONStorage._COMMIT_MAGIC

old_version = {'_default': {'1': {'version': 'old'}}}
new_version = {'_default': {'1': {'version': 'new'}}}


def frame(payload: bytes) -> bytes:
    """Build a valid journal frame for ``payload``."""
    if isinstance(payload, str):
        payload = payload.encode('utf-8')

    header = FRAME_MAGIC + struct.pack('>I', len(payload))
    checksum = zlib.crc32(header + payload) & 0xFFFFFFFF
    return header + payload + COMMIT_MAGIC + struct.pack('>I', checksum)


def dump_state(state):
    return json.dumps(state).encode('utf-8')


def make_main_file(path, state):
    with open(path, 'wb') as handle:
        handle.write(dump_state(state))


# ---------------------------------------------------------------------------
# Basic storage behaviour / interface compatibility
# ---------------------------------------------------------------------------

def test_is_storage_subclass():
    assert issubclass(JournalingJSONStorage, Storage)


def test_readwrite(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalingJSONStorage(path)

    assert storage.read() is None

    storage.write(old_version)
    assert storage.read() == old_version

    # Replace with a shorter document to make sure the main file is replaced,
    # not overwritten in place and left with trailing garbage.
    storage.write({'_default': {}})
    assert storage.read() == {'_default': {}}
    storage.close()

    # The state persists across a plain reopen
    storage = JournalingJSONStorage(path)
    assert storage.read() == {'_default': {}}
    storage.close()


def test_tinydb_integration(tmp_path):
    path = str(tmp_path / 'db.json')

    with TinyDB(path, storage=JournalingJSONStorage) as db:
        db.insert_multiple([{'name': 'first'}, {'name': 'second'}])
        assert len(db) == 2

    with TinyDB(path, storage=JournalingJSONStorage) as db:
        assert db.all() == [
            {'name': 'first'}, {'name': 'second'}
        ]
        assert db.contains(doc_id=2)


def test_readonly_mode(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    storage = JournalingJSONStorage(path, access_mode='r')
    assert storage.read() == old_version
    with pytest.raises(IOError):
        storage.write(new_version)
    storage.close()


def test_json_kwargs(tmp_path):
    path = str(tmp_path / 'db.json')

    with TinyDB(path, storage=JournalingJSONStorage, sort_keys=True, indent=1) as db:
        db.insert({'b': 1})
        db.insert({'a': 1})

    with open(path, encoding='utf-8') as handle:
        contents = handle.read()

    # Pretty-printed JSON (kwargs forwarded to json.dumps) and always valid
    assert json.loads(contents) == {
        '_default': {'1': {'b': 1}, '2': {'a': 1}}
    }
    assert '\n ' in contents


def test_encoding_is_preserved_across_recovery(tmp_path):
    path = str(tmp_path / 'db.json')
    japanese_doc = {'_default': {'1': {'greeting': 'こんにちは世界'}}}

    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(json.dumps(japanese_doc).encode('cp932')))

    storage = JournalingJSONStorage(path, encoding='cp932')
    assert storage.read() == japanese_doc
    storage.close()

    storage = JournalingJSONStorage(path, encoding='cp932')
    assert storage.read() == japanese_doc
    storage.close()


def test_caching_middleware_compatibility(tmp_path):
    path = str(tmp_path / 'db.json')

    db = TinyDB(path, storage=CachingMiddleware(JournalingJSONStorage))
    db.insert({'name': 'cached'})
    db.close()

    db = TinyDB(path, storage=CachingMiddleware(JournalingJSONStorage))
    assert db.all() == [{'name': 'cached'}]
    db.close()


def test_no_leftover_files(tmp_path):
    path = str(tmp_path / 'db.json')
    storage = JournalingJSONStorage(path)
    storage.write(old_version)
    storage.write(new_version)
    storage.close()

    assert os.listdir(str(tmp_path)) == ['db.json']


# ---------------------------------------------------------------------------
# Deterministic recovery from manipulated on-disk journal states
# ---------------------------------------------------------------------------

def reopen_payload(path):
    storage = JournalingJSONStorage(path)
    data = storage.read()
    storage.close()
    return data


def test_recovers_committed_journal(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)
    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(new_version)))

    assert reopen_payload(path) == new_version

    # Journal consumed; another reopen yields the same state deterministically
    assert reopen_payload(path) == new_version
    assert not os.path.exists(path + '.journal')


def test_truncated_payload_is_ignored(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    full = frame(dump_state(new_version))
    # Cut the frame in the middle of the payload
    with open(path + '.journal', 'wb') as handle:
        handle.write(full[: JournalingJSONStorage._HEADER_SIZE + 3])

    assert reopen_payload(path) == old_version
    assert not os.path.exists(path + '.journal')


def test_truncated_commit_boundary_is_ignored(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    full = frame(dump_state(new_version))
    # Payload fully on disk, but the commit boundary/trailer is missing
    with open(path + '.journal', 'wb') as handle:
        handle.write(full[:-5])

    assert reopen_payload(path) == old_version
    assert not os.path.exists(path + '.journal')


def test_checksum_failure_is_ignored(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    full = bytearray(frame(dump_state(new_version)))
    # Flip a byte inside the payload; the checksum must reject the frame
    full[JournalingJSONStorage._HEADER_SIZE + 2] ^= 0xFF
    with open(path + '.journal', 'wb') as handle:
        handle.write(full)

    assert reopen_payload(path) == old_version


def test_wrong_commit_marker_is_ignored(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    full = bytearray(frame(dump_state(new_version)))
    full[-8:-4] = b'XXXX'
    with open(path + '.journal', 'wb') as handle:
        handle.write(full)

    assert reopen_payload(path) == old_version


def test_garbage_journal_is_ignored(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    with open(path + '.journal', 'wb') as handle:
        handle.write(b'this is not a journal at all')

    assert reopen_payload(path) == old_version


def test_recovers_last_complete_commit(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    intermediate = {'_default': {'1': {'version': 'middle'}}}

    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(intermediate)))
        handle.write(frame(dump_state(new_version)))

    assert reopen_payload(path) == new_version


def test_last_complete_commit_beats_trailing_torn_frame(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)

    intermediate = {'_default': {'1': {'version': 'middle'}}}

    torn = frame(dump_state(new_version))
    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(intermediate)))
        handle.write(torn[:-8])  # complete frame, then a torn next frame

    assert reopen_payload(path) == intermediate


def test_recovery_is_idempotent(tmp_path):
    path = str(tmp_path / 'db.json')
    make_main_file(path, old_version)
    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(new_version)))

    for _ in range(3):
        storage = JournalingJSONStorage(path)
        assert storage.read() == new_version
        storage.close()

    assert not os.path.exists(path + '.journal')
    with open(path, 'rb') as handle:
        assert json.loads(handle.read()) == new_version


def test_recover_from_empty_database(tmp_path):
    path = str(tmp_path / 'db.json')
    # Main file empty (never written), committed journal present
    open(path, 'wb').close()
    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(new_version)))

    assert reopen_payload(path) == new_version


def test_checkpoint_then_crash_keeps_new_version(tmp_path):
    # The main file already contains the new state but journal removal was
    # interrupted. Reapplying the same state must be harmless.
    path = str(tmp_path / 'db.json')
    make_main_file(path, new_version)
    with open(path + '.journal', 'wb') as handle:
        handle.write(frame(dump_state(new_version)))

    assert reopen_payload(path) == new_version
    assert not os.path.exists(path + '.journal')


# ---------------------------------------------------------------------------
# Subprocess crash tests: a real process dies at each phase of a write
# ---------------------------------------------------------------------------

CHILD_SCRIPT = r"""
import os
import sys
import time

from tinydb.storages import JournalingJSONStorage

path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else None

storage = JournalingJSONStorage(path)

# First, durably commit a complete baseline ("old") version.
storage.write({'_default': {'1': {'version': 'old'}}})
storage.close()


def die(point):
    # Map a "hold" mode onto the write point at which the process should
    # block until the parent kills it with SIGKILL.
    hold_points = {'hold_frame': 'frame', 'hold_commit': 'commit'}
    hold_point = hold_points.get(mode)

    if hold_point is not None and point == hold_point:
        # Signal readiness to the parent process, then wait for SIGKILL.
        open(path + '.ready', 'w').close()
        while True:
            time.sleep(1)

    if mode is not None and point == mode:
        # Terminate immediately, without running any cleanup (atexit,
        # context managers, __del__), like SIGKILL.
        os._exit(7)

storage = JournalingJSONStorage(path, on_crash_point=die)
storage.write({'_default': {'1': {'version': 'new'}}})
storage.close()
"""


def run_crashed_writer(path, crash_point):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')

    argv = [sys.executable, '-c', CHILD_SCRIPT, path]
    if crash_point is not None:
        argv.append(crash_point)

    return subprocess.run(
        argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_sigkilled_writer(path, hold_point):
    """Start a writer that blocks at ``hold_point`` and kill it with SIGKILL."""
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')

    ready_path = path + '.ready'

    proc = subprocess.Popen(
        [sys.executable, '-c', CHILD_SCRIPT, path, hold_point],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.time() + 15
    while not os.path.exists(ready_path) and time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(proc.stderr.read().decode())
        time.sleep(0.01)

    assert os.path.exists(ready_path), 'writer never reached ' + hold_point

    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=15)
    assert proc.returncode == -signal.SIGKILL
    os.remove(ready_path)


@pytest.mark.parametrize('crash_point', [
    'journal_open',
    'header',
    'frame',
    'commit',
    'checkpoint',
])
def test_subprocess_crash_at_each_write_phase(tmp_path, crash_point):
    path = str(tmp_path / 'db.json')

    result = run_crashed_writer(path, crash_point)
    assert result.returncode == 7, result.stderr.decode()

    # Reopen in a fresh process (separate address space, cold caches)
    storage = JournalingJSONStorage(path)
    recovered = storage.read()
    storage.close()

    if crash_point in ('commit', 'checkpoint'):
        # 'commit': the commit boundary was fully durable but the main file
        # had not been replaced yet, so recovery replays the journal.
        # 'checkpoint': the main file was atomically replaced before the
        # process died during journal cleanup.
        assert recovered == new_version
    else:
        # Dying before the commit boundary is durable leaves the old version
        assert recovered == old_version

    # In every case the main file must contain a complete JSON document...
    with open(path, 'rb') as handle:
        main_contents = handle.read()
    assert json.loads(main_contents) in (old_version, new_version)

    # ...and reopening again must give the identical, deterministic result
    storage = JournalingJSONStorage(path)
    assert storage.read() == recovered
    storage.close()

    assert not os.path.exists(path + '.journal')
    assert not os.path.exists(path + '.tmp')


def test_subprocess_clean_run_then_reopen(tmp_path):
    path = str(tmp_path / 'db.json')

    result = run_crashed_writer(path, None)
    assert result.returncode == 0, result.stderr.decode()

    storage = JournalingJSONStorage(path)
    assert storage.read() == new_version
    storage.close()


@pytest.mark.parametrize('hold_point,expected', [
    # Payload on disk but no commit boundary -> previous version survives
    ('hold_frame', old_version),
    # Commit boundary durable but main file not replaced yet -> new version
    ('hold_commit', new_version),
])
def test_sigkill_during_write(tmp_path, hold_point, expected):
    path = str(tmp_path / 'db.json')

    run_sigkilled_writer(path, hold_point)

    storage = JournalingJSONStorage(path)
    assert storage.read() == expected
    storage.close()

    with open(path, 'rb') as handle:
        assert json.loads(handle.read()) == expected

    assert not os.path.exists(path + '.journal')


def test_subprocess_reopen_keeps_working_after_recovery(tmp_path):
    path = str(tmp_path / 'db.json')

    # First writer dies after the payload but before the commit boundary
    result = run_crashed_writer(path, 'frame')
    assert result.returncode == 7

    # Recovery happens here, then a new writer updates the database normally
    storage = JournalingJSONStorage(path)
    assert storage.read() == old_version
    storage.write(new_version)
    assert storage.read() == new_version
    storage.close()

    storage = JournalingJSONStorage(path)
    assert storage.read() == new_version
    storage.close()


def test_subprocess_reader_does_not_mutate_files(tmp_path):
    path = str(tmp_path / 'db.json')

    # A durable commit boundary but no checkpoint yet
    result = run_crashed_writer(path, 'commit')
    assert result.returncode == 7

    journal_before = open(path + '.journal', 'rb').read()

    storage = JournalingJSONStorage(path, access_mode='r')
    assert storage.read() == new_version
    storage.close()

    # Read-only access surfaces the committed journal state but leaves files
    # untouched; a subsequent writable open performs the actual recovery.
    assert open(path + '.journal', 'rb').read() == journal_before
    with open(path, 'rb') as handle:
        assert json.loads(handle.read()) == old_version

    storage = JournalingJSONStorage(path)
    assert storage.read() == new_version
    storage.close()
