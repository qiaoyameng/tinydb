"""
Optional composite indexes used by :class:`tinydb.table.Table`.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
import numbers
from typing import Any, Optional


class _Missing:
    def __repr__(self):
        return '<missing>'


_missing = _Missing()


def _safe_equal(left: Any, right: Any) -> bool:
    try:
        return left == right
    except TypeError:
        return False


def _range_bucket(value: Any) -> Optional[object]:
    if value is None:
        return type(None)
    if isinstance(value, numbers.Real):
        try:
            nan = math.isnan(value)
        except TypeError:
            nan = False
        if nan:
            return 'nan'
        return 'real'
    return type(value)


class _IndexNode:
    __slots__ = (
        'children', 'unhashable_children', 'document_ids',
        'range_entries', 'sorted_entries', 'unsorted_buckets'
    )

    def __init__(self) -> None:
        self.children: dict[Any, '_IndexNode'] = {}
        self.unhashable_children: list[tuple[Any, '_IndexNode']] = []
        self.document_ids: set[int] = set()
        self.range_entries: dict[object, list[tuple[Any, int]]] = {}
        self.sorted_entries: dict[object, list[tuple[Any, int]]] = {}
        self.unsorted_buckets: set[object] = set()

    def child(self, value: Any) -> Optional['_IndexNode']:
        try:
            hash(value)
        except TypeError:
            hashable = False
        else:
            hashable = True

        for existing_value, child in self.unhashable_children:
            if _safe_equal(existing_value, value):
                return child

        if hashable:
            try:
                for existing_value, child in self.children.items():
                    if _safe_equal(existing_value, value):
                        return child
            except RuntimeError:
                pass

        return None

    def get_or_create_child(self, value: Any) -> tuple['_IndexNode', bool]:
        child = self.child(value)
        if child is not None:
            return child, False

        child = _IndexNode()
        try:
            hash(value)
        except TypeError:
            self.unhashable_children.append((value, child))
        else:
            self.children[value] = child
        return child, True

    def remove_child_value(self, value: Any, child: '_IndexNode') -> None:
        try:
            hash(value)
        except TypeError:
            self.unhashable_children = [
                item for item in self.unhashable_children if item[1] is not child
            ]
        else:
            self.children = {
                key: existing for key, existing in self.children.items()
                if existing is not child
            }

    def add_range_entry(self, value: Any, doc_id: int) -> None:
        bucket = _range_bucket(value)
        if bucket is None:
            return

        entries = self.range_entries.setdefault(bucket, [])
        entries.append((value, doc_id))
        self.unsorted_buckets.add(bucket)

    def remove_range_entry(self, value: Any, doc_id: int) -> None:
        bucket = _range_bucket(value)
        if bucket is None or bucket == 'nan':
            return

        entries = self.range_entries.get(bucket)
        if entries is None:
            return

        entries[:] = [
            entry for entry in entries
            if entry[1] != doc_id or not _safe_equal(entry[0], value)
        ]
        if not entries:
            del self.range_entries[bucket]
            self.sorted_entries.pop(bucket, None)
        self.unsorted_buckets.add(bucket)

    def sorted_range(
            self, bucket: object) -> Optional[list[tuple[Any, int]]]:
        if bucket == 'nan':
            return []

        if bucket in self.unsorted_buckets:
            entries = self.range_entries.get(bucket, [])
            try:
                sorted_entries = sorted(entries, key=lambda item: (item[0], item[1]))
            except TypeError:
                return None
            self.sorted_entries[bucket] = sorted_entries
            self.unsorted_buckets.discard(bucket)

        return self.sorted_entries.get(bucket, [])


class CompositeIndex:
    """
    A composite index over top-level document fields.
    """

    def __init__(self, fields: tuple[str, ...]):
        if not fields or any(not isinstance(field, str) or not field
                             for field in fields):
            raise ValueError('Composite index fields must be non-empty strings')
        if len(set(fields)) != len(fields):
            raise ValueError('Composite index fields must be unique')

        self.fields = fields
        self._root = _IndexNode()

    def add(self, doc_id: int, values: tuple[Any, ...]) -> None:
        node = self._root
        nodes: list[_IndexNode] = []

        for value in values:
            if value is _missing:
                break
            node, _ = node.get_or_create_child(value)
            node.document_ids.add(doc_id)
            nodes.append(node)

        if len(nodes) == len(values):
            range_owner = self._root if len(nodes) == 1 else nodes[-2]
            range_owner.add_range_entry(values[-1], doc_id)

    def remove(self, doc_id: int, values: tuple[Any, ...]) -> None:
        node = self._root
        nodes: list[tuple[Any, _IndexNode]] = []

        for value in values:
            if value is _missing:
                break
            child = node.child(value)
            if child is None:
                return
            nodes.append((value, child))
            node = child

        if len(nodes) == len(values):
            range_owner = self._root if len(nodes) == 1 else nodes[-2][1]
            range_owner.remove_range_entry(values[-1], doc_id)

        for position in range(len(nodes) - 1, -1, -1):
            value, current = nodes[position]
            current.document_ids.discard(doc_id)
            parent = self._root if position == 0 else nodes[position - 1][1]
            if not current.document_ids:
                parent.remove_child_value(value, current)

    def _navigate(self, values: tuple[Any, ...]) -> Optional[_IndexNode]:
        node = self._root
        for value in values:
            child = node.child(value)
            if child is None:
                return None
            node = child
        return node

    def equality_candidates(self, values: tuple[Any, ...]) -> set[int]:
        node = self._navigate(values)
        return set() if node is None else set(node.document_ids)

    def range_candidates(
            self,
            prefix_values: tuple[Any, ...],
            operator: str,
            value: Any) -> Optional[set[int]]:
        node = self._navigate(prefix_values)
        if node is None:
            return set()

        if value is None:
            return None

        bucket = _range_bucket(value)
        present_buckets = set(node.range_entries)
        if any(present_bucket != bucket for present_bucket in present_buckets):
            return None

        if bucket == 'nan':
            entries = node.sorted_range(bucket)
            return None if entries is None else set()

        entries = node.sorted_range(bucket)
        if entries is None:
            return None

        minimum = -(2 ** 63)
        maximum = 2 ** 63 - 1
        if operator == '<':
            end = bisect_left(entries, (value, minimum))
            selected = entries[:end]
        elif operator == '<=':
            end = bisect_right(entries, (value, maximum))
            selected = entries[:end]
        elif operator == '>':
            start = bisect_right(entries, (value, maximum))
            selected = entries[start:]
        else:
            start = bisect_left(entries, (value, minimum))
            selected = entries[start:]

        return {doc_id for _, doc_id in selected}


@dataclass
class IndexPlan:
    index: CompositeIndex
    equality_values: tuple[Any, ...]
    range_value: Optional[tuple[str, Any]] = None

    def candidate_ids(self) -> Optional[set[int]]:
        if self.range_value is None:
            return self.index.equality_candidates(self.equality_values)

        operator, value = self.range_value
        return self.index.range_candidates(
            self.equality_values, operator, value
        )
