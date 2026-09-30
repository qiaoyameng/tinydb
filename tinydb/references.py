"""
This module implements optional referential constraints between tables.

A reference declares that a field of the documents in one table points to
the ``doc_id`` of a document in another (or the same) table. Before any
database write is committed, the complete candidate state is validated
against all registered references: missing targets either trigger the
configured delete behaviour (``restrict``, ``set_null`` or ``cascade``)
or cause the whole operation to be rejected, so a failed operation never
persists a partial result.
"""

from typing import Any, Optional

__all__ = (
    'RESTRICT',
    'SET_NULL',
    'CASCADE',
    'ON_DELETE_ACTIONS',
    'Reference',
    'ReferentialIntegrityError',
    'enforce_references',
    'validate_references',
)

#: Reject the operation if it would delete a still-referenced document
RESTRICT = 'restrict'

#: Set the referencing field to ``None`` when the target document is deleted
SET_NULL = 'set_null'

#: Delete the referencing document when the target document is deleted
CASCADE = 'cascade'

#: All supported delete behaviours
ON_DELETE_ACTIONS = (RESTRICT, SET_NULL, CASCADE)


class ReferentialIntegrityError(ValueError):
    """
    Raised when a write operation would violate a referential constraint.

    The offending operation is aborted before anything is written to the
    storage, so the database is left in its previous consistent state.
    """


class Reference:
    """
    A referential constraint between two tables.

    Declares that ``field`` of the documents in ``table`` stores the
    ``doc_id`` of a document in ``ref_table``. A field value of ``None``
    (or a missing field) means "no reference" and is always allowed.

    :param table: the name of the table holding the referencing field
    :param field: the document field that stores the target ``doc_id``
    :param ref_table: the name of the table the field points to
    :param on_delete: what happens when a referenced document is deleted,
                      one of ``RESTRICT``, ``SET_NULL`` or ``CASCADE``
    """

    __slots__ = ('table', 'field', 'ref_table', 'on_delete')

    def __init__(
        self,
        table: str,
        field: str,
        ref_table: str,
        on_delete: str = RESTRICT,
    ):
        if on_delete not in ON_DELETE_ACTIONS:
            raise ValueError(
                f'Invalid on_delete behaviour: {on_delete!r}, '
                f'expected one of {ON_DELETE_ACTIONS}'
            )

        self.table = table
        self.field = field
        self.ref_table = ref_table
        self.on_delete = on_delete

    def __repr__(self):
        return (
            f'{type(self).__name__}(table={self.table!r}, '
            f'field={self.field!r}, ref_table={self.ref_table!r}, '
            f'on_delete={self.on_delete!r})'
        )

    def __eq__(self, other):
        if not isinstance(other, Reference):
            return NotImplemented

        return (
            self.table == other.table
            and self.field == other.field
            and self.ref_table == other.ref_table
            and self.on_delete == other.on_delete
        )


def _dangling_documents(
    state: dict[str, dict[str, Any]],
    reference: Reference,
    target: dict[str, Any],
):
    """
    Yield ``(doc_id, document)`` pairs of the reference's source table
    whose referencing field points to a document missing from ``target``.
    """

    source = state.get(reference.table) or {}

    for doc_id, doc in source.items():
        value = doc.get(reference.field)

        # A missing or ``None`` field means "no reference"
        if value is None:
            continue

        if str(value) not in target:
            yield doc_id, doc


def validate_references(
    state: dict[str, dict[str, Any]],
    references,
) -> None:
    """
    Validate that all references in the given database state are intact.

    :raises ReferentialIntegrityError: if a referencing field points to a
        document that does not exist in the target table
    """

    for reference in references:
        target = state.get(reference.ref_table) or {}

        for doc_id, doc in _dangling_documents(state, reference, target):
            raise ReferentialIntegrityError(
                f'Document {doc_id} in table {reference.table!r} '
                f'references missing document '
                f'{doc[reference.field]!r} in table '
                f'{reference.ref_table!r} via field '
                f'{reference.field!r}'
            )


def enforce_references(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    references,
) -> None:
    """
    Enforce referential constraints on a candidate database state.

    ``before`` is the consistent database state before the pending
    operation, ``after`` is the candidate state produced by it. Documents
    whose reference target disappeared between the two states are treated
    according to the reference's ``on_delete`` behaviour, which is applied
    to ``after`` (modified in place) until a fixed point is reached.
    Cascades that encounter a reference cycle terminate because every
    document is deleted at most once.

    :raises ReferentialIntegrityError: if the candidate state cannot be
        made consistent. ``after`` may be partially modified in this case
        and must be discarded without being written to the storage.
    """

    # Apply delete behaviours until no more changes occur. Each iteration
    # either deletes at least one document or nulls at least one field, so
    # this terminates even for cyclic references.
    changed = True
    while changed:
        changed = False

        for reference in references:
            source = after.get(reference.table) or {}
            target = after.get(reference.ref_table) or {}
            target_before = before.get(reference.ref_table) or {}

            for doc_id, doc in list(source.items()):
                value = doc.get(reference.field)

                if value is None:
                    continue

                key = str(value)

                if key in target:
                    # The reference is intact
                    continue

                if key not in target_before:
                    # The target never existed, so this is an invalid
                    # reference (e.g. from an insert), not a deletion.
                    # ``validate_references`` below will reject it.
                    continue

                # The referenced document was deleted by this operation
                if reference.on_delete == RESTRICT:
                    raise ReferentialIntegrityError(
                        f'Cannot delete document {value!r} in table '
                        f'{reference.ref_table!r}: still referenced by '
                        f'document {doc_id} in table '
                        f'{reference.table!r} via field '
                        f'{reference.field!r}'
                    )
                elif reference.on_delete == CASCADE:
                    del source[doc_id]
                    changed = True
                else:  # SET_NULL
                    doc[reference.field] = None
                    changed = True

    # Reject any remaining dangling references (e.g. freshly inserted
    # documents pointing to documents that don't exist)
    validate_references(after, references)
