"""
Optional composite indexes for tables.
"""

from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any, Optional

from .storages import Storage


__all__ = ('CompositeIndex', 'IndexManager')


META_DOC_ID = '__tinydb_indexes__'
_RANGE_OPS = frozenset(('<', '<=', '>', '>='))
_SUPPORTED_OPS = frozenset(('==',)).union(_RANGE_OPS)


class _IndexValue:
    __slots__ = ('value', '_hash')

    def __init__(self, value: Any):
        self.value = value
        self._hash = hash(self._hashable(value))

    @staticmethod
    def _hashable(value: Any):
        if value is None:
            return ('__none__',)
        if isinstance(value, Mapping):
            return (
                '__mapping__',
                tuple(
                    (_IndexValue(key)._hash, _IndexValue(val)._hash)
                    for key, val in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(_IndexValue(item)._hash for item in value))
        if isinstance(value, (set, frozenset)):
            return ('__set__', frozenset(_IndexValue(item)._hash for item in value))
        try:
            return ('__value__', value)
        except TypeError:
            return ('__repr__', type(value).__module__, type(value).__qualname__, repr(value))

    def __eq__(self, other: object):
        if not isinstance(other, _IndexValue):
            return NotImplemented

        return bool(self.value == other.value)

    def __hash__(self):
        return self._hash


class CompositeIndex:
    def __init__(self, fields: tuple[tuple[str, ...], ...]):
        self.fields = fields
        self.entries: dict[tuple[_IndexValue, ...], set[Any]] = {}

    def key_for(self, document: Mapping) -> Optional[tuple[_IndexValue, ...]]:
        values = []
        for path in self.fields:
            value = document
            try:
                for part in path:
                    value = value[part]
            except (KeyError, TypeError):
                return None
            values.append(_IndexValue(value))
        return tuple(values)

    def add(self, key: tuple[_IndexValue, ...], doc_id: Any) -> None:
        self.entries.setdefault(key, set()).add(doc_id)

    def remove(self, key: tuple[_IndexValue, ...], doc_id: Any) -> None:
        ids = self.entries.get(key)
        if ids is None:
            return
        ids.discard(doc_id)
        if not ids:
            del self.entries[key]

    def clear(self) -> None:
        self.entries.clear()

    def rebuild(self, table: Mapping) -> None:
        self.clear()
        for doc_id, document in table.items():
            key = self.key_for(document)
            if key is not None:
                self.add(key, doc_id)


class IndexPlan:
    __slots__ = ('index', 'prefix', 'range_op', 'range_value')

    def __init__(
            self,
            index: CompositeIndex,
            prefix: tuple[_IndexValue, ...],
            range_op: Optional[str],
            range_value: Any):
        self.index = index
        self.prefix = prefix
        self.range_op = range_op
        self.range_value = range_value


class IndexTransaction:
    def __init__(self, manager: 'IndexManager'):
        self.manager = manager
        self.inserted: set[Any] = set()
        self.updated: set[Any] = set()
        self.removed: set[Any] = set()
        self.truncated = False
        self.old_keys: dict[Any, dict[CompositeIndex, Optional[tuple[_IndexValue, ...]]]] = {}

    def insert(self, doc_id: Any) -> None:
        self.inserted.add(doc_id)

    def update(self, table: Mapping, doc_id: Any) -> None:
        if doc_id not in self.old_keys:
            self.old_keys[doc_id] = self.manager.keys_for(table[doc_id])
        self.updated.add(doc_id)

    def remove(self, table: Mapping, doc_id: Any) -> None:
        if doc_id not in self.old_keys:
            self.old_keys[doc_id] = self.manager.keys_for(table[doc_id])
        self.removed.add(doc_id)

    def truncate(self) -> None:
        self.truncated = True

    def commit(self, table: Mapping) -> None:
        self.manager.apply_transaction(self, table)


class IndexManager:
    def __init__(
            self,
            storage: Storage,
            name: str,
            table: Mapping,
            metadata: Any = None,
            document_id_class: Callable[[Any], Any] = int):
        self._storage = storage
        self._name = name
        self._document_id_class = document_id_class
        self._indexes: dict[tuple[tuple[str, ...], ...], CompositeIndex] = {}
        self._revision = 0
        self._persist_metadata = False
        self._metadata_dirty = False
        self._load(table, metadata)

    @property
    def revision(self) -> int:
        return self._revision

    def indexes(self) -> tuple[CompositeIndex, ...]:
        return tuple(self._indexes.values())

    def create_index(self, *fields: Any) -> CompositeIndex:
        paths = self._normalize_fields(fields)
        existing = self._indexes.get(paths)
        if existing is not None:
            return existing

        index = CompositeIndex(paths)
        self._indexes[paths] = index
        index.rebuild(self._read_table())
        self._persist_metadata = True
        self._revision += 1
        self._metadata_dirty = True
        self._write_metadata()
        return index

    def drop_index(self, *fields: Any) -> None:
        paths = self._normalize_fields(fields)
        if paths not in self._indexes:
            return
        del self._indexes[paths]
        self._persist_metadata = True
        self._revision += 1
        self._metadata_dirty = True
        self._write_metadata()

    def transaction(self) -> IndexTransaction:
        return IndexTransaction(self)

    def apply_transaction(self, transaction: IndexTransaction, table: Mapping) -> None:
        if transaction.truncated:
            for index in self._indexes.values():
                index.clear()
        else:
            changed = transaction.inserted | transaction.updated
            for doc_id in transaction.removed - changed:
                old_keys = transaction.old_keys.get(doc_id, {})
                for index, key in old_keys.items():
                    if key is not None:
                        index.remove(key, doc_id)

            for doc_id in changed:
                if doc_id not in table:
                    continue
                old_keys = transaction.old_keys.get(doc_id)
                if old_keys is not None:
                    for index, key in old_keys.items():
                        if key is not None:
                            index.remove(key, doc_id)
                for index in self._indexes.values():
                    key = index.key_for(table[doc_id])
                    if key is not None:
                        index.add(key, doc_id)

        self._revision += 1
        self._metadata_dirty = True

    def keys_for(
            self,
            document: Mapping
    ) -> dict[CompositeIndex, Optional[tuple[_IndexValue, ...]]]:
        return {
            index: index.key_for(document)
            for index in self._indexes.values()
        }

    def attach_metadata(self, tables: dict[str, dict[str, Any]]) -> None:
        if not self._persist_metadata:
            return

        table = tables.setdefault(self._name, {})
        table[META_DOC_ID] = self._metadata(self._documents(table))
        self._metadata_dirty = False

    def plan_for(self, cond: Any) -> Optional[IndexPlan]:
        for index in self._indexes.values():
            plan = self._plan_for_index(cond, index)
            if plan is not None:
                return plan
        return None

    def lookup(self, plan: IndexPlan) -> set[Any]:
        result: set[Any] = set()
        prefix_length = len(plan.prefix)
        for key, ids in plan.index.entries.items():
            if key[:prefix_length] != plan.prefix:
                continue

            if plan.range_op is None:
                result.update(ids)
            else:
                if self._range_matches(
                        key[prefix_length].value,
                        plan.range_op,
                        plan.range_value
                ):
                    result.update(ids)

        return result

    @staticmethod
    def _range_matches(value: Any, op: str, rhs: Any) -> bool:
        if op == '<':
            return value < rhs
        if op == '<=':
            return value <= rhs
        if op == '>':
            return value > rhs
        return value >= rhs

    def _plan_for_index(self, cond: Any, index: CompositeIndex) -> Optional[IndexPlan]:
        leaves = self._flatten_and(cond)
        if leaves is None or not leaves or len(leaves) > len(index.fields):
            return None

        paths = tuple(getattr(leaf, '_query_path', None) for leaf in leaves)
        if paths != index.fields[:len(leaves)]:
            return None

        ops = tuple(getattr(leaf, '_query_op', None) for leaf in leaves)
        last_op = ops[-1]
        equal_count = len(leaves) if last_op == '==' else len(leaves) - 1
        if any(op != '==' for op in ops[:equal_count]):
            return None

        allowed_last = _SUPPORTED_OPS if len(leaves) == len(index.fields) else frozenset(('==',))
        if last_op not in allowed_last:
            return None

        prefix = tuple(
            _IndexValue(getattr(leaf, '_query_rhs'))
            for leaf in leaves[:equal_count]
        )
        range_op = last_op if last_op in _RANGE_OPS else None
        range_value = getattr(leaves[-1], '_query_rhs') if range_op is not None else None
        return IndexPlan(index, prefix, range_op, range_value)

    @staticmethod
    def _flatten_and(cond: Any) -> Optional[list[Any]]:
        op = getattr(cond, '_query_op', None)
        children = getattr(cond, '_children', None)
        if op == 'and' and children:
            result = []
            for child in children:
                leaves = IndexManager._flatten_and(child)
                if leaves is None:
                    return None
                result.extend(leaves)
            return result

        path = getattr(cond, '_query_path', None)
        if (
            op in _SUPPORTED_OPS
            and isinstance(path, tuple)
            and path
            and all(isinstance(part, str) for part in path)
        ):
            return [cond]
        return None

    @staticmethod
    def _normalize_fields(fields: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
        if not fields:
            raise ValueError('Composite index requires at least one field')

        paths = []
        for field in fields:
            if isinstance(field, str):
                path = (field,)
            elif (
                isinstance(field, tuple)
                and field
                and all(isinstance(part, str) and part for part in field)
            ):
                path = field
            else:
                raise ValueError(
                    'Composite index fields must be strings or '
                    'non-empty string tuples'
                )
            if path in paths:
                raise ValueError('Composite index fields must be unique')
            paths.append(path)
        return tuple(paths)

    def _load(self, table: Mapping, metadata: Any) -> None:
        if not isinstance(metadata, Mapping):
            return

        self._persist_metadata = True
        try:
            stored_revision = int(metadata['revision'])
            definitions = [
                self._normalize_fields(tuple(tuple(path) for path in definition))
                for definition in metadata['indexes']
            ]
            stored_digest = metadata['digest']
        except (KeyError, TypeError, ValueError):
            self._revision = 1
            return

        for paths in definitions:
            index = CompositeIndex(paths)
            index.rebuild(table)
            self._indexes[paths] = index

        digest = self._digest(table) if definitions or 'digest' in metadata else None
        self._revision = stored_revision
        if digest != stored_digest:
            self._revision += 1
            self._metadata_dirty = True

    def _metadata(self, table: Mapping) -> dict[str, Any]:
        return {
            'revision': self._revision,
            'indexes': [
                [list(path) for path in index.fields]
                for index in self._indexes.values()
            ],
            'digest': self._digest(table),
        }

    def _write_metadata(self) -> None:
        tables = self._storage.read()
        if tables is None:
            tables = {}
        table = tables.setdefault(self._name, {})
        table[META_DOC_ID] = self._metadata(self._documents(table))
        self._storage.write(tables)
        self._metadata_dirty = False

    def _read_table(self) -> Mapping:
        tables = self._storage.read()
        if tables is None:
            return {}
        table = tables.get(self._name, {})
        return self._documents(table)

    def _documents(self, table: Mapping) -> dict[Any, Mapping]:
        return {
            self._document_id_class(doc_id): document
            for doc_id, document in table.items()
            if doc_id != META_DOC_ID
        }

    def _digest(self, table: Mapping) -> str:
        digest = sha256()
        digest.update(str(len(self._indexes)).encode())
        for index in self._indexes.values():
            for path in index.fields:
                digest.update(repr(path).encode())
        for doc_id in sorted(table.keys(), key=_stable_sort_key):
            digest.update(b'\x00')
            digest.update(str(doc_id).encode())
            digest.update(_stable_bytes(table[doc_id]))
        return digest.hexdigest()


def _stable_sort_key(value: Any):
    return type(value).__module__, type(value).__qualname__, repr(value)


def _stable_bytes(value: Any) -> bytes:
    marker = f'{type(value).__module__}.{type(value).__qualname__}:'.encode()
    if isinstance(value, Mapping):
        items = sorted(
            value.items(),
            key=lambda item: (
                type(item[0]).__module__,
                type(item[0]).__qualname__,
                repr(item[0]),
            )
        )
        return marker + b'{' + b','.join(
            _stable_bytes(key) + b':' + _stable_bytes(val) for key, val in items
        ) + b'}'
    if isinstance(value, list):
        return marker + b'[' + b','.join(_stable_bytes(item) for item in value) + b']'
    if isinstance(value, tuple):
        return marker + b'(' + b','.join(_stable_bytes(item) for item in value) + b')'
    if isinstance(value, (set, frozenset)):
        items = sorted(value, key=_stable_sort_key)
        return marker + b'{' + b','.join(_stable_bytes(item) for item in items) + b'}'
    return marker + repr(value).encode()
