"""
Contains the :class:`base class <tinydb.storages.Storage>` for storages and
implementations.
"""

import io
import json
import os
import struct
import warnings
import zlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional

__all__ = ('Storage', 'JSONStorage', 'JournalingJSONStorage', 'MemoryStorage')


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


def _fsync_directory(path: str) -> None:
    """
    Fsync the directory that contains ``path``'s file.

    This is required to make the creation or removal of a file (and a rename)
    durable. On platforms that do not support fsyncing directories the call
    is silently ignored.
    """

    directory = os.path.dirname(os.path.abspath(path))

    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except (OSError, AttributeError):
        return

    try:
        os.fsync(dir_fd)
    except (OSError, AttributeError):
        # Some platforms (e.g. Windows) do not allow fsyncing directories
        pass
    finally:
        os.close(dir_fd)


class JournalingJSONStorage(Storage):
    """
    Store the data in a JSON file, protected by a write-ahead journal.

    Unlike :class:`JSONStorage`, this storage survives a process that is
    killed (or crashes, or loses power) while a write is in progress. Before
    the main data file is replaced, the complete new state is appended to a
    journal file together with a checksum and an explicit commit boundary.
    Every step is flushed and ``fsync``-ed, so durability does not depend on
    :meth:`close` being called.

    When the storage is opened it first replays the journal:

    * a journal entry whose commit boundary or checksum is missing (for
      example because the file was truncated mid-write) is ignored, and
    * the last fully committed entry is restored into the main data file.

    As a result, after reopening, the database is guaranteed to contain
    either the previous complete version or the new complete version, never
    a partially written JSON document.

    Recovery is idempotent: once a committed entry has been applied to the
    main data file, the journal is removed; opening the storage again simply
    reads the main data file.

    .. warning::
        The main data file and its journal must be accessed by at most one
        writer process or thread at a time. Concurrent writers may lose
        updates or corrupt the journal. Concurrent read-only access while a
        single writer is running is safe.

    The journal is stored next to the main file in a file named
    ``<path>.journal``.
    """

    #: Magic bytes that mark the beginning of a journal frame.
    _FRAME_MAGIC = b'TJNL'

    #: Magic bytes that mark a successfully committed journal frame.
    _COMMIT_MAGIC = b'COMT'

    #: Size in bytes of a frame's header (magic + payload length).
    _HEADER_SIZE = 8

    #: Size in bytes of a frame's trailer (commit marker + checksum).
    _TRAILER_SIZE = 8

    def __init__(self, path: str, create_dirs=False, encoding=None,
                 access_mode='r+', on_crash_point=None, **kwargs):
        """
        Create a new instance.

        Also creates the storage file, if it doesn't exist and the access
        mode is appropriate for writing. Any journal left behind by a
        previous process is replayed before the storage becomes available.

        :param path: Where to store the JSON data. A journal is kept next to
                     it in a file named ``<path>.journal``.
        :param create_dirs: Whether to create all missing parent
                            directories.
        :param encoding: The encoding used for the JSON data.
        :param access_mode: Mode in which the main file is opened
                            (``r``/``rb`` for read-only, ``r+``/``rb+`` for
                            reading and writing).
        :param on_crash_point: Optional callable invoked at well-defined
                               points of a write (``'journal_open'``,
                               ``'header'``, ``'frame'``, ``'commit'``,
                               ``'checkpoint'``, ``'after'``). It is meant
                               for testing crash recovery and is not part of
                               the stable API.
        :param kwargs: Extra keyword arguments forwarded to ``json.dumps``.
        """

        super().__init__()

        self._mode = access_mode
        self._encoding = encoding or 'utf-8'
        self.kwargs = kwargs
        self._on_crash_point = on_crash_point

        if access_mode not in ('r', 'rb', 'r+', 'rb+'):
            warnings.warn(
                'Using an `access_mode` other than \'r\', \'rb\', \'r+\' '
                'or \'rb+\' can cause data loss or corruption'
            )

        self._writable = any(character in self._mode for character in ('+', 'w', 'a'))

        self._path = path
        self._journal_path = path + '.journal'
        self._temp_path = path + '.tmp'

        if self._writable:
            touch(path, create_dirs=create_dirs)
            self._recover_if_needed()

        # Open the main file for reading/writing
        self._handle = open(path, mode=self._mode, encoding=encoding)

    def close(self) -> None:
        self._handle.close()

    @classmethod
    def _read_journal_payloads(cls, journal_path: str,
                               encoding: str) -> Iterator[bytes]:
        """
        Parse the journal file and yield the payload of every complete,
        checksum-valid frame in order.

        Parsing stops deterministically at the first incomplete or invalid
        frame (e.g. a truncated commit boundary or a checksum mismatch);
        everything after it is ignored.
        """

        try:
            handle = open(journal_path, 'rb')
        except OSError:
            return

        with handle:
            while True:
                header = handle.read(cls._HEADER_SIZE)
                if not header:
                    # Clean end of the journal
                    return
                if len(header) < cls._HEADER_SIZE or \
                        header[:4] != cls._FRAME_MAGIC:
                    return

                (length,) = struct.unpack('>I', header[4:])

                payload = handle.read(length)
                trailer = handle.read(cls._TRAILER_SIZE)

                if len(payload) < length or len(trailer) < cls._TRAILER_SIZE:
                    # Truncated payload or missing/partial commit boundary
                    return

                if trailer[:4] != cls._COMMIT_MAGIC:
                    return

                (checksum,) = struct.unpack('>I', trailer[4:])
                if (zlib.crc32(header + payload) & 0xFFFFFFFF) != checksum:
                    # Corrupted payload or header
                    return

                try:
                    json.loads(payload.decode(encoding))
                except (UnicodeDecodeError, ValueError):
                    # Corrupted payload that happened to pass the checksum
                    return

                yield payload

    def _recover_if_needed(self) -> None:
        """
        Replay the journal and restore the last complete commit.

        After recovery the journal is removed, so reopening the storage
        repeatedly yields the same, deterministic result.
        """

        last_payload = None
        for payload in self._read_journal_payloads(self._journal_path,
                                                   self._encoding):
            last_payload = payload

        if last_payload is not None:
            self._checkpoint(last_payload)

        # A leftover temp file can only stem from an interrupted checkpoint;
        # the main file is still the previous committed version in that case.
        if os.path.exists(self._temp_path):
            os.remove(self._temp_path)
            _fsync_directory(self._path)

        if os.path.exists(self._journal_path):
            # Contains no complete frame, or the last commit was applied:
            # discard it so the result is stable across reopens.
            os.remove(self._journal_path)
            _fsync_directory(self._path)

    def _checkpoint(self, payload: bytes) -> None:
        """
        Durably replace the main data file with ``payload``.
        """

        with open(self._temp_path, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(self._temp_path, self._path)
        _fsync_directory(self._path)

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        # A journal may exist if this instance was opened read-only while a
        # crashed writer left a committed transaction behind. The committed
        # journal entry always represents the newest complete state.
        journal_payload = None
        for payload in self._read_journal_payloads(self._journal_path,
                                                   self._encoding):
            journal_payload = payload

        if journal_payload is not None:
            return json.loads(journal_payload.decode(self._encoding))

        self._handle.seek(0, os.SEEK_END)
        size = self._handle.tell()

        if not size:
            return None

        self._handle.seek(0)
        return json.load(self._handle)

    def write(self, data: dict[str, dict[str, Any]]) -> None:
        if not self._writable:
            raise IOError(
                'Cannot write to the database. Access mode is "{0}"'.format(
                    self._mode))

        payload = json.dumps(data, **self.kwargs).encode(self._encoding)

        journal_existed = os.path.exists(self._journal_path)

        with open(self._journal_path, 'ab') as journal:
            if self._on_crash_point is not None:
                self._on_crash_point('journal_open')

            header = self._FRAME_MAGIC + struct.pack('>I', len(payload))
            checksum = zlib.crc32(header + payload) & 0xFFFFFFFF
            trailer = self._COMMIT_MAGIC + struct.pack('>I', checksum)

            journal.write(header)
            journal.flush()
            os.fsync(journal.fileno())
            if not journal_existed:
                # Make the creation of the journal itself durable
                _fsync_directory(self._journal_path)

            if self._on_crash_point is not None:
                self._on_crash_point('header')

            journal.write(payload)
            journal.flush()
            os.fsync(journal.fileno())

            if self._on_crash_point is not None:
                self._on_crash_point('frame')

            # The commit boundary is written last: its presence and the
            # checksum make this frame an all-or-nothing commit.
            journal.write(trailer)
            journal.flush()
            os.fsync(journal.fileno())

        if self._on_crash_point is not None:
            self._on_crash_point('commit')

        # Replace the main data file with the committed state. The main file
        # itself is never overwritten in place, so it always holds a complete
        # version of the data.
        self._checkpoint(payload)

        if self._on_crash_point is not None:
            self._on_crash_point('checkpoint')

        os.remove(self._journal_path)
        _fsync_directory(self._path)

        # The main file was replaced via os.replace(); reopen our handle so
        # it points at the new inode.
        self._handle.close()
        self._handle = open(self._path, mode=self._mode,
                            encoding=self._encoding)

        if self._on_crash_point is not None:
            self._on_crash_point('after')


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
