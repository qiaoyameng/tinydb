"""
Optional referential integrity constraints between TinyDB tables.

A :class:`Reference` declares that a field of the documents in a source
table points to the ``doc_id`` of a document in a target table. References
are optional: documents that don't contain the field or store ``None`` in
it never violate the constraint.

Constraints are enforced against the full candidate state of the database
*before* a write is committed. If enforcement fails, a
:class:`ReferentialIntegrityError` is raised and the storage is left
untouched, so a failed operation never commits a partial result.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = (
    'Reference',
    'ReferentialIntegrityError',
    'RESTRICT',
    'SET_NULL',
    'CASCADE',
    'ON_DELETE_ACTIONS',
    'enforce_references',
)

#: Delete behavior: reject the operation while references to the removed
#: documents exist
RESTRICT = 'restrict'

#: Delete behavior: reset the referencing field to ``None``
SET_NULL = 'set_null'

#: Delete behavior: delete the referencing documents as well
CASCADE = 'cascade'

#: All supported ``on_delete`` behaviors
ON_DELETE_ACTIONS = (RESTRICT, SET_NULL, CASCADE)


class ReferentialIntegrityError(ValueError):
    """
    Raised when a write operation would violate a reference constraint.

    The database is left unchanged when this error is raised: constraints
    are checked against the full candidate state before anything is
    committed to the storage.
    """


@dataclass(frozen=True)
class Reference:
    """
    A referential constraint between two tables.

    Documents of ``source_table`` may store the ``doc_id`` of a document
    of ``target_table`` in ``field``. The reference is optional: a missing
    field or a ``None`` value is always valid.

    ``on_delete`` controls what happens when a referenced document is
    removed (via ``remove``, ``truncate`` or ``drop_table``):

    - ``RESTRICT``: the operation is rejected,
    - ``SET_NULL``: the referencing field is reset to ``None``,
    - ``CASCADE``: the referencing documents are deleted as well
      (recursively; cycles are handled by processing every document at
      most once).
    """

    source_table: str
    field: str
    target_table: str
    on_delete: str = RESTRICT

    def __post_init__(self):
        if self.on_delete not in ON_DELETE_ACTIONS:
            raise ValueError(
                f'Invalid on_delete behavior: {self.on_delete!r}. '
                f'Expected one of: {", ".join(ON_DELETE_ACTIONS)}'
            )


def _referenced_id(document: dict, field: str) -> Optional[str]:
    """
    Return the referenced doc ID (as a string) or ``None`` if the document
    doesn't reference anything.
    """

    value = document.get(field)

    if value is None:
        return None

    return str(value)


def enforce_references(
    tables: dict,
    references: Iterable[Reference],
    removed: Iterable[tuple] = (),
) -> dict:
    """
    Apply referential actions and validate all constraints in-place.

    ``tables`` is the full candidate state of the database *after* the
    user's operation has been applied. ``removed`` lists the
    ``(table_name, doc_id)`` pairs that the operation deleted, which
    triggers the ``on_delete`` behaviors.

    The candidate state is mutated in place (cascading removals and
    ``set_null`` resets) and returned. If any constraint is violated, a
    :class:`ReferentialIntegrityError` is raised; callers must then
    discard the candidate state instead of committing it.

    :param tables: the candidate database state
    :param references: the constraints to enforce
    :param removed: doc IDs deleted by the triggering operation
    :returns: the enforced candidate state
    """

    references = list(references)

    if not references:
        return tables

    # Normalize the removed IDs to strings so they can be compared with
    # reference field values regardless of the storage's key type.
    removed_ids = {(table, str(doc_id)) for table, doc_id in removed}

    # 1. Compute the cascade closure. Every document is processed at most
    #    once, so cyclic cascade graphs terminate instead of looping.
    cascade_refs = [ref for ref in references if ref.on_delete == CASCADE]
    queue = list(removed_ids)

    while queue:
        table_name, doc_id = queue.pop()

        for ref in cascade_refs:
            if ref.target_table != table_name:
                continue

            source = tables.get(ref.source_table)

            if not source:
                continue

            for source_id, document in list(source.items()):
                if (ref.source_table, str(source_id)) in removed_ids:
                    # Already removed: don't touch it again (cycle dedup)
                    continue

                if _referenced_id(document, ref.field) == doc_id:
                    del source[source_id]
                    removed_ids.add((ref.source_table, str(source_id)))
                    queue.append((ref.source_table, str(source_id)))

    # 2. Reset references to removed documents for ``set_null`` constraints.
    for ref in references:
        if ref.on_delete != SET_NULL:
            continue

        removed_in_target = {
            doc_id for table, doc_id in removed_ids
            if table == ref.target_table
        }

        if not removed_in_target:
            continue

        source = tables.get(ref.source_table)

        if not source:
            continue

        for document in source.values():
            referenced = _referenced_id(document, ref.field)

            if referenced is not None and referenced in removed_in_target:
                document[ref.field] = None

    # 3. Validate the full candidate state: every non-null reference must
    #    point to an existing document. This covers ``restrict`` deletions
    #    as well as inserts/updates of dangling references.
    for ref in references:
        source = tables.get(ref.source_table)

        if not source:
            continue

        target = tables.get(ref.target_table) or {}
        target_ids = {str(doc_id) for doc_id in target.keys()}

        for source_id, document in source.items():
            referenced = _referenced_id(document, ref.field)

            if referenced is not None and referenced not in target_ids:
                raise ReferentialIntegrityError(
                    f'Reference constraint violated: '
                    f'{ref.source_table}.{ref.field} -> '
                    f'{ref.target_table}: document {source_id} references '
                    f'missing doc_id {referenced}'
                )

    return tables
