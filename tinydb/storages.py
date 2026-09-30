"""
Contains the :class:`base class <tinydb.storages.Storage>` for storages and
implementations.
"""

import hashlib
import io
import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ('Storage', 'JSONStorage', 'MemoryStorage', 'JournalStorage')


# On-disk format of the journal used by :class:`JournalStorage`.
#
# A fully committed journal is a UTF-8 compatible binary file with exactly
# this layout::
#
#     TINYDB-JOURNAL-V1\n          <- header / format version
#     <sha256-hex> <length>\n     <- payload checksum and length
#     <payload bytes>              <- exact bytes of the new main data file
#     COMMIT\n                     <- commit boundary / record
#
# Only a journal that is complete (the declared number of payload bytes
# are present), ends with exactly the commit record and whose SHA-256
# checksum matches is considered a valid commit. Everything else is
# discarded during recovery, leaving the previous main data file intact.
_JOURNAL_HEADER = b'TINYDB-JOURNAL-V1\n'
_JOURNAL_COMMIT = b'COMMIT\n'
_JOURNAL_METADATA_RE = re.compile(
    r'(?P<digest>[0-9a-f]{64}) (?P<length>[0-9]+)'
)


def touch(path: str, create_dirs: bool):
    """
    Create a file if it doesn't exist yet.

    :param path: The file to create.
    :param create_dirs: Whether to create all missing parent directories.
    """
    if create_dirs:
        base_dir = os.path.dirname(path)

        # Check if we need to create missing parent directories
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

    # Create the file by opening it in 'a' mode which creates the file if it
    # does not exist yet but does not modify its contents
    with open(path, 'a'):
        pass


class Storage(ABC):
    """
    The abstract base class for all Storages.

    A Storage (de)serializes the current state of the database and stores it in
    some place (memory, file on disk, ...).
    """

    # Using ABCMeta as metaclass allows instantiating only storages that have
    # implemented read and write

    @abstractmethod
    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        """
        Read the current state.

        Any kind of deserialization should go here.

        Return ``None`` here to indicate that the storage is empty.
        """

        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def write(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Write the current state of the database to the storage.

        Any kind of serialization should go here.

        :param data: The current state of the database.
        """

        raise NotImplementedError('To be overridden!')

    def close(self) -> None:
        """
        Optional: Close open file handles, etc.
        """

        pass


class JSONStorage(Storage):
    """
    Store the data in a JSON file.
    """

    def __init__(self, path: str, create_dirs=False, encoding=None, access_mode='r+', **kwargs):
        """
        Create a new instance.

        Also creates the storage file, if it doesn't exist and the access mode
        is appropriate for writing.

        **Note:** Using an access mode other than `r` or `r+` will probably
        lead to data loss or data corruption!

        **Note:** **Never** pass untrusted or user-controlled code as ``kwargs``
        members like ``cls`` or ``default`` will be called on every write
        operation.

        :param path: Where to store the JSON data.
        :param access_mode: mode in which the file is opened (r, r+)
        :type access_mode: str
        """

        super().__init__()

        self._mode = access_mode
        self.kwargs = kwargs

        if access_mode not in ('r', 'rb', 'r+', 'rb+'):
            warnings.warn(
                'Using an `access_mode` other than \'r\', \'rb\', \'r+\' '
                'or \'rb+\' can cause data loss or corruption'
            )

        # Create the file if it doesn't exist and creating is allowed by the
        # access mode
        if any([character in self._mode for character in ('+', 'w', 'a')]):  # any of the writing modes
            touch(path, create_dirs=create_dirs)

        # Open the file for reading/writing
        self._handle = open(path, mode=self._mode, encoding=encoding)

    def close(self) -> None:
        self._handle.close()

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        # Get the file size by moving the cursor to the file end and reading
        # its location
        self._handle.seek(0, os.SEEK_END)
        size = self._handle.tell()

        if not size:
            # File is empty, so we return ``None`` so TinyDB can properly
            # initialize the database
            return None
        else:
            # Return the cursor to the beginning of the file
            self._handle.seek(0)

            # Load the JSON contents of the file
            return json.load(self._handle)

    def write(self, data: dict[str, dict[str, Any]]):
        # Move the cursor to the beginning of the file just in case
        self._handle.seek(0)

        # Serialize the database state using the user-provided arguments
        serialized = json.dumps(data, **self.kwargs)

        # Write the serialized data to the file
        try:
            self._handle.write(serialized)
        except io.UnsupportedOperation:
            raise IOError('Cannot write to the database. Access mode is "{0}"'.format(self._mode))

        # Ensure the file has been written
        self._handle.flush()
        os.fsync(self._handle.fileno())

        # Remove data that is behind the new cursor in case the file has
        # gotten shorter
        self._handle.truncate()


class JournalStorage(Storage):
    """
    Store the data in a JSON file using a write-ahead journal.

    This storage is a drop-in replacement for :class:`JSONStorage` that keeps
    the database file consistent even when the process is killed (or the
    machine loses power) while a write is in progress. After reopening, the
    database contains either the state before the interrupted write or the
    complete new state -- never a truncated or otherwise half-written JSON
    file.

    Every write goes through the following stages, with an ``fsync`` at every
    step that must survive a crash:

    1. The new state is serialized and written, together with a SHA-256
       checksum and its length, to a temporary journal file
       (``<path>.journal.tmp``).
    2. The temporary journal is atomically renamed to ``<path>.journal``.
    3. A commit record (``COMMIT``) is appended to the journal. A journal is
       considered committed only if it ends with this record.
    4. The journal payload is written to a temporary data file
       (``<path>.tmp``) which is then atomically renamed to the main data
       file.
    5. The journal is removed.

    On initialization any pending journal is recovered:

    - a complete, checksum-valid journal is installed as the main data file
      (the last complete commit wins),
    - a missing, truncated or checksum-invalid journal is discarded and the
      previous main data file is kept,
    - stale temporary files from interrupted writes are removed.

    Recovery is deterministic and idempotent: reopening the storage multiple
    times has the same result as reopening it once.

    Data is flushed and ``fsync``'ed as part of every :meth:`write` call; it
    is not deferred until :meth:`close`.

    .. admonition:: Single writer

       :class:`JournalStorage` does not perform any inter-process locking. A
       given data file (including its journal) must only be written by a
       single process (and a single storage instance) at a time. Multiple
       readers are safe as long as no writer is active concurrently.
    """

    def __init__(self, path: str, create_dirs=False, encoding=None,
                 access_mode='r+', **kwargs):
        """
        Create a new instance.

        The constructor performs crash recovery before the storage can be
        used: a pending complete commit is installed and every other kind of
        leftover journal or temporary file is removed.

        :param path: Where to store the JSON data.
        :param create_dirs: Whether to create all missing parent
                            directories.
        :param encoding: Encoding used for the JSON data file (defaults to
                         UTF-8 for the journal payload).
        :param access_mode: Mode in which the data file is opened (r, rb,
                            r+, rb+). In a read-only mode (``r``/``rb``) no
                            recovery is performed and writes raise
                            ``IOError``.
        """

        super().__init__()

        self._mode = access_mode
        self.kwargs = kwargs
        self._file_encoding = encoding
        self._encoding = encoding or 'utf-8'
        self._read_only = access_mode in ('r', 'rb')

        self.path = path
        self._journal_path = path + '.journal'
        self._journal_tmp_path = path + '.journal.tmp'
        self._main_tmp_path = path + '.tmp'

        if access_mode not in ('r', 'rb', 'r+', 'rb+'):
            warnings.warn(
                'Using an `access_mode` other than \'r\', \'rb\', \'r+\' '
                'or \'rb+\' can cause data loss or corruption'
            )

        if not self._read_only:
            # Create the file if it doesn't exist and creating is allowed by
            # the access mode, then replay or discard a pending journal.
            touch(path, create_dirs=create_dirs)
            self._recover()

        # Open the file for reading/writing
        self._handle = open(path, mode=self._mode, encoding=encoding)

    def close(self) -> None:
        self._handle.close()

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        # Get the file size by moving the cursor to the file end and reading
        # its location
        self._handle.seek(0, os.SEEK_END)
        size = self._handle.tell()

        if not size:
            # File is empty, so we return ``None`` so TinyDB can properly
            # initialize the database
            return None

        # Return the cursor to the beginning of the file and load the JSON
        self._handle.seek(0)
        return json.load(self._handle)

    def write(self, data: dict[str, dict[str, Any]]) -> None:
        if self._read_only:
            raise IOError(
                'Cannot write to the database. Access mode is "{0}"'.format(
                    self._mode
                )
            )

        # Serialize the database state using the user-provided arguments
        serialized = json.dumps(data, **self.kwargs)
        payload = serialized.encode(self._encoding)
        digest = hashlib.sha256(payload).hexdigest()

        # Write-ahead journal: everything is flushed and fsync'ed before the
        # main data file is replaced, so a crash at any point leaves either
        # the old state or a recoverable complete commit.
        self._stage_journal(payload, digest)
        self._publish_journal()
        self._commit_journal()
        self._install_payload(payload)
        self._discard_journal()

        # The atomic replace swapped the main data file for a new inode, so
        # reopen the file handle to keep reading and writing working.
        self._handle.close()
        self._handle = open(
            self.path, mode=self._mode, encoding=self._file_encoding
        )

    def _recover(self) -> None:
        """
        Restore the last complete commit, if any, and clean up leftovers.

        This is safe to run repeatedly: it either installs a valid committed
        journal or removes any invalid/incomplete journal and temporary
        files. The main data file is never modified in the latter case.
        """

        # Remove temporary files an interrupted write may have left behind
        for tmp_path in (self._journal_tmp_path, self._main_tmp_path):
            self._remove_file(tmp_path)

        payload = self._read_journal()
        if payload is not None:
            # The journal contains a complete commit: install it
            self._install_payload(payload)

        # Valid journals have been installed, invalid/truncated ones are
        # discarded. Removing the journal is the end of recovery.
        self._remove_file(self._journal_path)
        self._fsync_directory()

    def _read_journal(self) -> Optional[bytes]:
        """
        Return the journal payload if and only if the journal is a valid
        complete commit, otherwise ``None``.
        """

        try:
            handle = open(self._journal_path, 'rb')
        except FileNotFoundError:
            return None

        with handle:
            if handle.readline() != _JOURNAL_HEADER:
                return None

            metadata = handle.readline().decode(
                'ascii', errors='ignore'
            ).strip()
            match = _JOURNAL_METADATA_RE.fullmatch(metadata)
            if match is None:
                return None

            digest = match.group('digest')
            length = int(match.group('length'))

            payload = handle.read(length)
            if len(payload) != length:
                # Truncated payload
                return None

            if handle.read() != _JOURNAL_COMMIT:
                # Missing, partial or trailing data instead of the commit
                # boundary
                return None

        if hashlib.sha256(payload).hexdigest() != digest:
            # Checksum verification failed
            return None

        return payload

    def _stage_journal(self, payload: bytes, digest: str) -> None:
        """
        Write header, checksum metadata and payload to the journal temp file.
        """

        metadata = '{0} {1}\n'.format(digest, len(payload)).encode('ascii')
        with open(self._journal_tmp_path, 'wb') as handle:
            handle.write(_JOURNAL_HEADER)
            handle.write(metadata)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _publish_journal(self) -> None:
        """
        Atomically move the staged journal into its final name.
        """

        os.replace(self._journal_tmp_path, self._journal_path)
        self._fsync_directory()

    def _commit_journal(self) -> None:
        """
        Append the commit boundary, making the journal a complete commit.
        """

        with open(self._journal_path, 'ab') as handle:
            handle.write(_JOURNAL_COMMIT)
            handle.flush()
            os.fsync(handle.fileno())

    def _install_payload(self, payload: bytes) -> None:
        """
        Write the payload to a temporary data file and atomically replace the
        main data file with it.
        """

        with open(self._main_tmp_path, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # The main data file may currently be opened through ``self._handle``.
        # Close it before the atomic replace: on Windows a file cannot be
        # replaced while it is open. On POSIX closing is harmless. The caller
        # is responsible for reopening the handle. During constructor-time
        # recovery the handle doesn't exist yet.
        if hasattr(self, '_handle') and not self._handle.closed:
            self._handle.close()

        os.replace(self._main_tmp_path, self.path)
        self._fsync_directory()

    def _discard_journal(self) -> None:
        """
        Remove the journal after the main data file has been replaced.
        """

        os.remove(self._journal_path)
        self._fsync_directory()

    @staticmethod
    def _remove_file(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _fsync_directory(self) -> None:
        """
        Fsync the directory so renames and unlinks are crash-durable.
        """

        directory = os.path.dirname(os.path.abspath(self.path))
        dir_fd: int | None = None
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            os.fsync(dir_fd)
        except OSError:
            # Some platforms (e.g. Windows) don't support fsync on directory
            # file descriptors
            pass  # pragma: no cover
        finally:
            if dir_fd is not None:
                os.close(dir_fd)


class MemoryStorage(Storage):
    """
    Store the data as JSON in memory.
    """

    def __init__(self):
        """
        Create a new instance.
        """

        super().__init__()
        self.memory = None

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        return self.memory

    def write(self, data: dict[str, dict[str, Any]]):
        self.memory = data
