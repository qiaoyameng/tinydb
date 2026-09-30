"""
Subprocess worker used by the journal storage crash tests.

It runs a single JournalStorage write while exiting hard at a defined stage
or while being slow enough for the parent process to send SIGKILL.
"""

import json
import os
import sys
import time

from tinydb.storages import (
    JournalStorage,
    _JOURNAL_COMMIT,
    _JOURNAL_HEADER,
)

OLD_DATA = {'_default': {'1': {'version': 'old', 'pad': 'x' * 64}}}
NEW_DATA = {'_default': {'1': {'version': 'new', 'pad': 'y' * 4096}}}

OLD_JSON = json.dumps(OLD_DATA)
NEW_JSON = json.dumps(NEW_DATA)


def _hard_exit():
    # Exit without flushing buffers or running cleanup, simulating a crash
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(7)


class CrashStorage(JournalStorage):
    """
    JournalStorage that exits the process hard once a given write stage has
    completed.
    """

    crash_phase: str | None = None

    def _stage_journal(self, payload, digest):
        super()._stage_journal(payload, digest)
        if self.crash_phase == 'staged':
            _hard_exit()

    def _publish_journal(self):
        super()._publish_journal()
        if self.crash_phase == 'published':
            _hard_exit()

    def _commit_journal(self):
        super()._commit_journal()
        if self.crash_phase == 'committed':
            _hard_exit()

    def _install_payload(self, payload):
        super()._install_payload(payload)
        if self.crash_phase == 'installed':
            _hard_exit()

    def _discard_journal(self):
        super()._discard_journal()
        if self.crash_phase == 'discarded':
            _hard_exit()


class SlowStorage(JournalStorage):
    """
    JournalStorage that sleeps inside every write stage so a SIGKILL can hit
    while any part of the write protocol is in progress.
    """

    def _sleep(self):
        time.sleep(0.05)

    def _stage_journal(self, payload, digest):
        metadata = '{0} {1}\n'.format(digest, len(payload)).encode('ascii')
        with open(self._journal_tmp_path, 'wb') as handle:
            handle.write(_JOURNAL_HEADER)
            self._sleep()
            handle.write(metadata)
            self._sleep()
            handle.write(payload)
            self._sleep()
            handle.flush()
            os.fsync(handle.fileno())
        self._sleep()

    def _publish_journal(self):
        self._sleep()
        super()._publish_journal()
        self._sleep()

    def _commit_journal(self):
        with open(self._journal_path, 'ab') as handle:
            self._sleep()
            handle.write(_JOURNAL_COMMIT)
            handle.flush()
            os.fsync(handle.fileno())
        self._sleep()

    def _install_payload(self, payload):
        with open(self._main_tmp_path, 'wb') as handle:
            self._sleep()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._sleep()
        os.replace(self._main_tmp_path, self.path)
        self._fsync_directory()
        self._sleep()

    def _discard_journal(self):
        self._sleep()
        super()._discard_journal()
        self._sleep()


def main():
    command = sys.argv[1]
    path = sys.argv[2]

    if command == 'init':
        storage = JournalStorage(path)
        storage.write(OLD_DATA)
        storage.close()
    elif command == 'crash-write':
        phase = sys.argv[3]
        CrashStorage.crash_phase = phase
        storage = CrashStorage(path)
        storage.write(NEW_DATA)
        storage.close()
    elif command == 'kill-write':
        storage = SlowStorage(path)
        storage.write(NEW_DATA)
        storage.close()
    elif command == 'read':
        storage = JournalStorage(path)
        data = storage.read()
        storage.close()
        sys.stdout.write(json.dumps(data))
        sys.stdout.write('\n')
    else:  # pragma: no cover
        raise SystemExit('unknown command: {0}'.format(command))


if __name__ == '__main__':
    main()
