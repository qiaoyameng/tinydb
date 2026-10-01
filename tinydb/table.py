"""
This module implements tables, the central place for accessing and manipulating
data in TinyDB.
"""

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
import hashlib
import json
from typing import (
    Any,
    NoReturn,
    Optional,
    Union,
    cast,
    overload
)

from .indexes import CompositeIndex, IndexPlan, _missing
from .queries import QueryLike
from .storages import Storage
from .utils import LRUCache

__all__ = ('Document', 'Table', 'CompositeIndex')

IndexDefinitions = Optional[Mapping[str, Union[str, Iterable[str]]]]


class Document(dict):
    """
    A document stored in the database.

    This class provides a way to access both a document's content and
    its ID using ``doc.doc_id``.
    """

    def __init__(self, value: Mapping, doc_id: int):
        super().__init__(value)
        self.doc_id = doc_id


class Table:
    """
    Represents a single TinyDB table.

    It provides methods for accessing and manipulating documents.

    .. admonition:: Query Cache

        As an optimization, a query cache is implemented using a
        :class:`~tinydb.utils.LRUCache`. This class mimics the interface of
        a normal ``dict``, but starts to remove the least-recently used entries
        once a threshold is reached.

        The query cache is updated on every search operation. When writing
        data, the whole cache is discarded as the query results may have
        changed.

    .. admonition:: Customization

        For customization, the following class variables can be set:

        - ``document_class`` defines the class that is used to represent
          documents,
        - ``document_id_class`` defines the class that is used to represent
          document IDs,
        - ``query_cache_class`` defines the class that is used for the query
          cache
        - ``default_query_cache_capacity`` defines the default capacity of
          the query cache

        .. versionadded:: 4.0


    :param storage: The storage instance to use for this table
    :param name: The table name
    :param cache_size: Maximum capacity of query cache
    :param persist_empty: Store new table even with no operations on it
    :param indexes: Composite indexes to create for this table
    """

    #: The class used to represent documents
    #:
    #: .. versionadded:: 4.0
    document_class = Document

    #: The class used to represent a document ID
    #:
    #: .. versionadded:: 4.0
    document_id_class = int

    #: The class used for caching query results
    #:
    #: .. versionadded:: 4.0
    query_cache_class = LRUCache

    #: The default capacity of the query cache
    #:
    #: .. versionadded:: 4.0
    default_query_cache_capacity = 10

    def __init__(
        self,
        storage: Storage,
        name: str,
        cache_size: int = default_query_cache_capacity,
        persist_empty: bool = False,
        indexes: IndexDefinitions = None,
    ):
        """
        Create a table instance.
        """

        self._storage = storage
        self._name = name
        self._query_cache: LRUCache[QueryLike, list[Document]] \
            = self.query_cache_class(capacity=cache_size)

        self._next_id: Optional[int] = None
        self._indexes: dict[str, CompositeIndex] = {}
        self._index_revision = 0
        self._index_metadata_dirty = False
        self._pending_index_definitions = dict(indexes or {})
        self._indexes_loaded = not self._pending_index_definitions
        self._index_check_pending = True

        if persist_empty:
            self._update_table(lambda table: table.clear())

    def __repr__(self):
        args = [
            'name={!r}'.format(self.name),
            'total={}'.format(len(self)),
            'storage={}'.format(self._storage),
        ]

        return '<{} {}>'.format(type(self).__name__, ', '.join(args))

    @property
    def name(self) -> str:
        """
        Get the table name.
        """
        return self._name

    @property
    def storage(self) -> Storage:
        """
        Get the table storage instance.
        """
        return self._storage

    def insert(self, document: Mapping) -> int:
        """
        Insert a new document into the table.

        :param document: the document to insert
        :returns: the inserted document's ID
        """

        # Make sure the document implements the ``Mapping`` interface
        if not isinstance(document, Mapping):
            raise ValueError('Document is not a Mapping')

        # First, we get the document ID for the new document
        if isinstance(document, self.document_class):
            # For a `Document` object we use the specified ID
            doc_id = document.doc_id

            # We also reset the stored next ID so the next insert won't
            # re-use document IDs by accident when storing an old value
            self._next_id = None
        else:
            # In all other cases we use the next free ID
            doc_id = self._get_next_id()

        # Now, we update the table and add the document
        def updater(table: dict):
            if doc_id in table:
                raise ValueError(f'Document with ID {str(doc_id)} '
                                 f'already exists')

            # By calling ``dict(document)`` we convert the data we got to a
            # ``dict`` instance even if it was a different class that
            # implemented the ``Mapping`` interface
            table[doc_id] = dict(document)

        # See below for details on ``Table._update``
        self._update_table(updater)

        return doc_id

    def insert_multiple(self, documents: Iterable[Mapping]) -> list[int]:
        """
        Insert multiple documents into the table.

        :param documents: an Iterable of documents to insert
        :returns: a list containing the inserted documents' IDs
        """
        doc_ids = []

        def updater(table: dict):
            for document in documents:

                # Make sure the document implements the ``Mapping`` interface
                if not isinstance(document, Mapping):
                    raise ValueError('Document is not a Mapping')

                if isinstance(document, self.document_class):
                    # Check if document does not override an existing document
                    if document.doc_id in table:
                        raise ValueError(
                            f'Document with ID {str(document.doc_id)} '
                            f'already exists'
                        )

                    # Store the doc_id, so we can return all document IDs
                    # later. Then save the document with its doc_id and
                    # skip the rest of the current loop
                    doc_id = document.doc_id
                    doc_ids.append(doc_id)
                    table[doc_id] = dict(document)
                    # Same as insert(): don't reuse this ID for later
                    # auto-assigned documents in this batch.
                    self._next_id = None
                    continue

                # Generate new document ID for this document.
                # Allocate from the in-progress table, not storage, so a
                # Document inserted earlier in this batch is visible.
                next_id = self._next_id
                if next_id is None:
                    if table:
                        next_id = max(
                            self.document_id_class(i) for i in table.keys()
                        ) + 1
                    else:
                        next_id = 1
                self._next_id = next_id
                doc_id = self._get_next_id()
                doc_ids.append(doc_id)
                table[doc_id] = dict(document)

        # See below for details on ``Table._update``
        self._update_table(updater)

        return doc_ids

    def all(self) -> list[Document]:
        """
        Get all documents stored in the table.

        :returns: a list with all documents.
        """

        # iter(self) (implemented in Table.__iter__ provides an iterator
        # that returns all documents in this table. We use it to get a list
        # of all documents by using the ``list`` constructor to perform the
        # conversion.

        return list(iter(self))

    def search(self, cond: QueryLike) -> list[Document]:
        """
        Search for all documents matching a 'where' cond.

        Note: a repeated search with the same ``cond`` is served from an
        internal cache. The returned list is a fresh copy, but its
        documents are the same objects held by the cache -- treat them as
        read-only, since mutating one (a key or a nested value) corrupts
        what a later identical search returns.

        :param cond: the condition to check against
        :returns: list of matching documents
        """

        # First, we check the query cache to see if it has results for this
        # query
        cached_results = self._query_cache.get(cond)
        if cached_results is not None:
            return cached_results[:]

        self._ensure_indexes_loaded()
        plan = self._index_plan(cond)
        docs = self._search_with_plan(cond, plan)

        # Only cache cacheable queries.
        #
        # This weird `getattr` dance is needed to make MyPy happy as
        # it doesn't know that a query might have a `is_cacheable` method
        # that is not declared in the `QueryLike` protocol due to it being
        # optional.
        # See: https://github.com/python/mypy/issues/1424
        #
        # Note also that by default we expect custom query objects to be
        # cacheable (which means they need to have a stable hash value).
        # This is to keep consistency with TinyDB's behavior before
        # `is_cacheable` was introduced which assumed that all queries
        # are cacheable.
        is_cacheable: Callable[[], bool] = getattr(cond, 'is_cacheable',
                                                   lambda: True)
        if is_cacheable():
            # Update the query cache
            self._query_cache[cond] = docs[:]

        return docs

    def _search_scan(self, cond: QueryLike) -> list[Document]:
        return [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in self._read_table().items()
            if cond(doc)
        ]

    def _search_with_plan(
            self, cond: QueryLike, plan: Optional[IndexPlan]) -> list[Document]:
        table = self._read_table()
        if plan is None:
            candidate_ids = None
        else:
            candidate_ids = plan.candidate_ids()

        return [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in table.items()
            if (candidate_ids is None
                or self.document_id_class(doc_id) in candidate_ids)
            and cond(doc)
        ]

    @overload
    def get(self) -> NoReturn: ...

    @overload
    def get(
        self, cond: QueryLike, doc_id: None = ..., doc_ids: None = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, *, cond: QueryLike, doc_id: None = ..., doc_ids: None = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], doc_id: int, doc_ids: Optional[list] = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, *, cond: Optional[QueryLike] = ..., doc_id: int, doc_ids: Optional[list] = ...,
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], doc_id: None, doc_ids: list
    ) -> list[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], *, doc_id: None = ..., doc_ids: list
    ) -> list[Document]: ...

    @overload
    def get(
        self, *, cond: Optional[QueryLike] = ..., doc_id: None = ..., doc_ids: list
    ) -> list[Document]: ...

    def get(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None,
        doc_ids: Optional[list] = None
    ):
        """
        Get exactly one document specified by a query or a document ID.
        However, if multiple document IDs are given then returns all
        documents in a list.
        
        Returns ``None`` if the document doesn't exist.

        :param cond: the condition to check against
        :param doc_id: the document's ID
        :param doc_ids: the document's IDs(multiple)

        :returns: the document(s) or ``None``
        """
        table = self._read_table()

        if doc_id is not None:
            self._ensure_indexes_loaded()
            # Retrieve a document specified by its ID
            raw_doc = table.get(str(doc_id), None)

            if raw_doc is None:
                return None

            # Convert the raw data to the document class
            return self.document_class(raw_doc, doc_id)

        elif doc_ids is not None:
            self._ensure_indexes_loaded()
            # Filter the table by extracting out all those documents which
            # have doc id specified in the doc_id list.

            # Since document IDs will be unique, we make it a set to ensure
            # constant time lookup
            doc_ids_set = set(str(doc_id) for doc_id in doc_ids)

            # Now return the filtered documents in form of list
            return [
                self.document_class(doc, self.document_id_class(doc_id))
                for doc_id, doc in table.items()
                if doc_id in doc_ids_set
            ]

        elif cond is not None:
            self._ensure_indexes_loaded()
            # Find a document specified by a query
            # The trailing underscore in doc_id_ is needed so MyPy
            # doesn't think that `doc_id_` (which is a string) needs
            # to have the same type as `doc_id` which is this function's
            # parameter and is an optional `int`.
            for doc_id_, doc in self._read_table().items():
                if cond(doc):
                    return self.document_class(
                        doc,
                        self.document_id_class(doc_id_)
                    )

            return None

        raise RuntimeError('You have to pass either cond or doc_id or doc_ids')

    def contains(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None
    ) -> bool:
        """
        Check whether the database contains a document matching a query or
        an ID.

        If ``doc_id`` is set, it checks if the db contains the specified ID.

        :param cond: the condition use
        :param doc_id: the document ID to look for
        """
        if doc_id is not None:
            # Documents specified by ID
            return self.get(doc_id=doc_id) is not None

        elif cond is not None:
            # Document specified by condition
            return self.get(cond) is not None

        raise RuntimeError('You have to pass either cond or doc_id')

    def update(
        self,
        fields: Union[Mapping, Callable[[MutableMapping], None]],
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """
        Update all matching documents to have a given set of fields.

        When ``doc_ids`` is given, IDs that don't refer to an existing
        document are silently skipped, and the returned list only contains
        the IDs that were actually updated.

        :param fields: the fields that the matching documents will have
                       or a callable that mutates matching documents in place
        :param cond: which documents to update
        :param doc_ids: a list of document IDs
        :returns: a list containing the updated document's ID
        """

        # Define the function that will perform the update
        if callable(fields):
            def perform_update(table, doc_id):
                # Update documents by calling the update function provided by
                # the user
                fields(table[doc_id])
        else:
            def perform_update(table, doc_id):
                # Update documents by setting all fields from the provided data
                table[doc_id].update(fields)

        if doc_ids is not None:
            # Perform the update operation for documents specified by a list
            # of document IDs. Document IDs that don't exist in the table are
            # silently skipped, mirroring the behaviour of ``get(doc_ids=...)``
            # (see issue #591). The list of *actually* updated IDs is
            # determined inside the updater so it reflects the table state at
            # write time.
            requested_ids = list(doc_ids)
            updated_ids: list[int] = []

            def updater(table: dict):
                # Filter to IDs that exist *before* performing any updates,
                # so the operation is atomic: either every existing target is
                # updated, or none is.
                updated_ids.extend(
                    doc_id for doc_id in requested_ids if doc_id in table
                )
                for doc_id in updated_ids:
                    perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            return updated_ids

        elif cond is not None:
            # Perform the update operation for documents specified by a query

            # Collect affected doc_ids
            updated_ids = []

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            return updated_ids

        else:
            # Update all documents unconditionally

            updated_ids = []

            def updater(table: dict):
                # Process all documents
                for doc_id in list(table.keys()):
                    # Add ID to list of updated documents
                    updated_ids.append(doc_id)

                    # Perform the update (see above)
                    perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            return updated_ids

    def update_multiple(
        self,
        updates: Iterable[
            tuple[Union[Mapping, Callable[[MutableMapping], None]], QueryLike]
        ],
    ) -> list[int]:
        """
        Update all matching documents to have a given set of fields.

        :returns: a list containing the updated document's ID
        """

        # Define the function that will perform the update
        def perform_update(fields, table, doc_id):
            if callable(fields):
                # Update documents by calling the update function provided
                # by the user
                fields(table[doc_id])
            else:
                # Update documents by setting all fields from the provided
                # data
                table[doc_id].update(fields)

        # Perform the update operation for documents specified by a query

        # Collect affected doc_ids
        updated_ids = []

        def updater(table: dict):
            # We need to convert the keys iterator to a list because
            # we may remove entries from the ``table`` dict during
            # iteration and doing this without the list conversion would
            # result in an exception (RuntimeError: dictionary changed size
            # during iteration)
            for doc_id in list(table.keys()):
                for fields, cond in updates:
                    _cond = cast(QueryLike, cond)

                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(fields, table, doc_id)

        # Perform the update operation (see _update_table for details)
        self._update_table(updater)

        return updated_ids

    def upsert(self, document: Mapping, cond: Optional[QueryLike] = None) -> list[int]:
        """
        Update documents, if they exist, insert them otherwise.

        Note: This will update *all* documents matching the query.
        For example, if the query matches 3 documents, all 3 will be updated
        with the new data. If no documents match, a new one is inserted.

        Document argument can be a tinydb.table.Document object if you want
        to specify a doc_id.

        :param document: the document to insert or the fields to update
        :param cond: which document to look for, optional if you've passed a
        Document with a doc_id
        :returns: a list containing the updated documents' IDs
        """

        # Extract doc_id
        if isinstance(document, self.document_class) and hasattr(document, 'doc_id'):
            doc_ids: Optional[list[int]] = [document.doc_id]
        else:
            doc_ids = None

        # Make sure we can actually find a matching document
        if doc_ids is None and cond is None:
            raise ValueError("If you don't specify a search query, you must "
                             "specify a doc_id. Hint: use a table.Document "
                             "object.")

        # Perform the update operation. ``update`` returns an empty list
        # when no existing document matches the doc_id / query, in which
        # case we fall through to insert below.
        updated_docs = self.update(document, cond, doc_ids)

        # If documents have been updated: return their IDs
        if updated_docs:
            return updated_docs

        # There are no documents that match the specified query -> insert the
        # data as a new document
        return [self.insert(document)]

    def remove(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """
        Remove all matching documents.

        When ``doc_ids`` is given, IDs that don't refer to an existing
        document are silently skipped, and the returned list only contains
        the IDs that were actually removed.

        :param cond: the condition to check against
        :param doc_ids: a list of document IDs
        :returns: a list containing the removed documents' ID
        """
        if doc_ids is not None:
            # This function returns the list of IDs for the documents that
            # have been removed. Document IDs that don't exist in the table
            # are silently skipped, mirroring the behaviour of
            # ``get(doc_ids=...)`` (see issue #591). The list of *actually*
            # removed IDs is determined inside the updater so it reflects the
            # table state at write time.
            requested_ids = list(doc_ids)
            removed_ids: list[int] = []

            def updater(table: dict):
                # Filter to IDs that exist *before* performing any removals,
                # so the operation is atomic: either every existing target is
                # removed, or none is.
                removed_ids.extend(
                    doc_id for doc_id in requested_ids if doc_id in table
                )
                for doc_id in removed_ids:
                    del table[doc_id]

            # Perform the remove operation
            self._update_table(updater)

            return removed_ids

        if cond is not None:
            removed_ids = []

            # This updater function will be called with the table data
            # as its first argument. See ``Table._update`` for details on this
            # operation
            def updater(table: dict):
                # We need to convince MyPy (the static type checker) that
                # the ``cond is not None`` invariant still holds true when
                # the updater function is called
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    if _cond(table[doc_id]):
                        # Add document ID to list of removed document IDs
                        removed_ids.append(doc_id)

                        # Remove document from the table
                        table.pop(doc_id)

            # Perform the remove operation
            self._update_table(updater)

            return removed_ids

        raise RuntimeError('Use truncate() to remove all documents')

    def truncate(self) -> None:
        """
        Truncate the table by removing all documents.
        """

        # Update the table by resetting all data
        self._update_table(lambda table: table.clear())

        # Reset document ID counter
        self._next_id = None

    def count(self, cond: QueryLike) -> int:
        """
        Count the documents matching a query.

        :param cond: the condition use
        """

        return len(self.search(cond))

    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """

        self._query_cache.clear()

    def _reset_indexes(self) -> None:
        self._indexes.clear()
        self._index_revision = 0
        self._index_metadata_dirty = False
        self._pending_index_definitions = {}
        self._indexes_loaded = True
        self._index_check_pending = False

    def indexes(self) -> dict[str, CompositeIndex]:
        """
        Get the table's composite indexes.
        """

        self._ensure_indexes_loaded()
        return dict(self._indexes)

    def add_index(
            self,
            fields: Union[str, Iterable[str]],
            name: Optional[str] = None) -> str:
        """
        Add and build a composite index.
        """

        self._ensure_indexes_loaded()
        normalized = self._normalize_index_fields(fields)
        index_name = name or 'idx_' + '_'.join(normalized)
        if not index_name:
            raise ValueError('Index name must not be empty')

        existing = self._indexes.get(index_name)
        if existing is not None and existing.fields != normalized:
            raise ValueError(f'Index {index_name!r} already exists')
        if existing is not None:
            return index_name

        index = CompositeIndex(normalized)
        for doc_id, document in self._read_table().items():
            index.add(self._index_doc_id(doc_id),
                      self._index_values(document, index))
        self._indexes[index_name] = index
        self._index_revision += 1

        tables = self._storage.read()
        if tables is None:
            tables = {}
        self._write_index_metadata(tables, self._read_table())
        self._storage.write(tables)
        self.clear_cache()
        return index_name

    def drop_index(self, name: str) -> None:
        """
        Remove a composite index.
        """

        self._ensure_indexes_loaded()
        if name not in self._indexes:
            return

        del self._indexes[name]
        tables = self._storage.read() or {}
        self._write_index_metadata(tables, self._read_table())
        self._storage.write(tables)
        self.clear_cache()

    def rebuild_index(self, name: str) -> None:
        """
        Rebuild a composite index from the current table data.
        """

        self._ensure_indexes_loaded()
        index = self._indexes[name]
        rebuilt = CompositeIndex(index.fields)
        for doc_id, document in self._read_table().items():
            rebuilt.add(self._index_doc_id(doc_id),
                        self._index_values(document, rebuilt))
        self._indexes[name] = rebuilt
        self._index_revision += 1

        tables = self._storage.read() or {}
        self._write_index_metadata(tables, self._read_table())
        self._storage.write(tables)
        self.clear_cache()

    def _ensure_indexes_loaded(
            self, tables: Optional[dict[str, Any]] = None) -> None:
        if self._indexes_loaded and not self._index_check_pending:
            return
        if tables is None and (
                self._index_check_pending or self._pending_index_definitions):
            tables = self._storage.read() or {}
        self._load_indexes(self._pending_index_definitions, tables)
        self._indexes_loaded = True
        self._index_check_pending = False

    def _load_indexes(
            self,
            definitions: Mapping[str, Union[str, Iterable[str]]],
            tables: Optional[dict[str, Any]] = None):
        if tables is None:
            tables = self._storage.read() or {}

        metadata = self._index_metadata_from_tables(tables)
        raw_table = tables.get(self.name, {})
        table_metadata = (tables.get('__tinydb_indexes__') or {}).get(self.name, {})
        if not isinstance(table_metadata, Mapping):
            table_metadata = {}
        table_revision = table_metadata.get('revision')
        table_checksum = table_metadata.get('checksum')
        revisions = []

        configured = {
            name: self._normalize_index_fields(fields)
            for name, fields in definitions.items()
        }
        for name, fields in configured.items():
            entry = metadata.get(name)
            if entry is not None and tuple(entry.get('fields', ())) == fields:
                if isinstance(table_revision, int):
                    revisions.append(table_revision)
            else:
                self._index_metadata_dirty = True

        for name, entry in metadata.items():
            fields = entry.get('fields')
            if not isinstance(fields, list) or not all(
                    isinstance(field, str) for field in fields):
                self._index_metadata_dirty = True
                continue

            try:
                index = CompositeIndex(tuple(fields))
            except ValueError:
                self._index_metadata_dirty = True
                continue

            for doc_id, document in raw_table.items():
                index.add(self._index_doc_id(doc_id),
                          self._index_values(document, index))
            self._indexes[name] = index

            if isinstance(table_revision, int):
                self._index_revision = table_revision
            if not isinstance(table_checksum, str) or \
                    table_checksum != self._index_checksum(raw_table):
                self._index_metadata_dirty = True

        for name, fields in configured.items():
            existing = self._indexes.get(name)
            if existing is None:
                index = CompositeIndex(fields)
                for doc_id, document in raw_table.items():
                    index.add(self._index_doc_id(doc_id),
                              self._index_values(document, index))
                self._indexes[name] = index
            elif existing.fields != fields:
                self.rebuild_index(name)

        if revisions:
            self._index_revision = max(revisions)

        if self._index_metadata_dirty and self._indexes:
            self._index_revision += 1
            if self.name not in tables:
                tables[self.name] = {}
            normalized_table = {
                self._index_doc_id(doc_id): document
                for doc_id, document in tables[self.name].items()
            }
            self._write_index_metadata(tables, normalized_table)
            self._storage.write(tables)

    @staticmethod
    def _flatten_and(query: Any) -> Optional[list[tuple]]:
        if not isinstance(query, tuple) or not query:
            return None
        if query[0] == 'and':
            left = Table._flatten_and(query[1])
            right = Table._flatten_and(query[2])
            if left is None or right is None:
                return None
            return left + right
        if query[0] in ('or', 'not'):
            return None
        if query[0] == 'cmp':
            return [query]
        return None

    def _index_plan(self, cond: QueryLike) -> Optional[IndexPlan]:
        self._ensure_indexes_loaded()
        expression = getattr(cond, '_query', None)
        comparisons = self._flatten_and(expression)
        if comparisons is None:
            return None

        predicates_by_field: dict[str, list[tuple[str, Any]]] = {}
        for _, operator, path, value in comparisons:
            if len(path) != 1 or not isinstance(path[0], str):
                return None
            predicates_by_field.setdefault(path[0], []).append(
                (operator, value)
            )

        for index in self._indexes.values():
            equality_values = []
            range_predicate = None
            valid = True
            covered_fields = set()

            for position, field in enumerate(index.fields):
                predicates = predicates_by_field.get(field, [])

                if position == len(index.fields) - 1:
                    if not predicates:
                        break
                    equality_predicates = [
                        predicate for predicate in predicates
                        if predicate[0] == '=='
                    ]
                    range_predicates = [
                        predicate for predicate in predicates
                        if predicate[0] in ('<', '<=', '>', '>=')
                    ]
                    if len(equality_predicates) == 1 and not range_predicates:
                        equality_values.append(equality_predicates[0][1])
                    elif not equality_predicates and len(range_predicates) == 1:
                        range_predicate = range_predicates[0]
                    else:
                        valid = False
                        break
                    covered_fields.add(field)
                else:
                    if not predicates:
                        break
                    if len(predicates) != 1 or predicates[0][0] != '==':
                        valid = False
                        break
                    equality_values.append(predicates[0][1])
                    covered_fields.add(field)

            if not valid:
                continue

            if set(predicates_by_field) != covered_fields:
                valid = False
            if not valid:
                continue

            return IndexPlan(
                index=index,
                equality_values=tuple(equality_values),
                range_value=None if range_predicate is None else (
                    range_predicate[0], range_predicate[1]
                ),
            )

        return None

    def __len__(self):
        """
        Count the total number of documents in this table.
        """

        return len(self._read_table())

    def __iter__(self) -> Iterator[Document]:
        """
        Iterate over all documents stored in the table.

        :returns: an iterator over all documents.
        """

        # Iterate all documents and their IDs
        for doc_id, doc in self._read_table().items():
            # Convert documents to the document class
            yield self.document_class(doc, self.document_id_class(doc_id))

    def _get_next_id(self):
        """
        Return the ID for a newly inserted document.
        """

        # If we already know the next ID
        if self._next_id is not None:
            next_id = self._next_id
            self._next_id = next_id + 1

            return next_id

        # Determine the next document ID by finding out the max ID value
        # of the current table documents

        # Read the table documents
        table = self._read_table()

        # If the table is empty, set the initial ID
        if not table:
            next_id = 1
            self._next_id = next_id + 1

            return next_id

        # Determine the next ID based on the maximum ID that's currently in use
        max_id = max(self.document_id_class(i) for i in table.keys())
        next_id = max_id + 1

        # The next ID we will return AFTER this call needs to be larger than
        # the current next ID we calculated
        self._next_id = next_id + 1

        return next_id

    def _read_table(self) -> dict[str, Mapping]:
        """
        Read the table data from the underlying storage.

        Documents and doc_ids are NOT yet transformed, as
        we may not want to convert *all* documents when returning
        only one document for example.
        """

        # Retrieve the tables from the storage
        tables = self._storage.read()
        if self._index_check_pending:
            self._prime_index_check(tables)

        if tables is None:
            # The database is empty
            return {}

        # Retrieve the current table's data
        try:
            table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            return {}

        return table

    def _prime_index_check(self, tables: Optional[dict[str, Any]]) -> None:
        if not self._pending_index_definitions:
            root = (tables or {}).get('__tinydb_indexes__')
            if not isinstance(root, Mapping) or self.name not in root:
                self._indexes_loaded = True
                self._index_check_pending = False

    def _update_table(self, updater: Callable[[dict[int, Mapping]], None]):
        """
        Perform a table update operation.

        The storage interface used by TinyDB only allows to read/write the
        complete database data, but not modifying only portions of it. Thus,
        to only update portions of the table data, we first perform a read
        operation, perform the update on the table data and then write
        the updated data back to the storage.

        As a further optimization, we don't convert the documents into the
        document class, as the table data will *not* be returned to the user.
        """

        tables = self._storage.read()
        self._ensure_indexes_loaded(tables)

        if tables is None:
            # The database is empty
            tables = {}

        try:
            raw_table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            raw_table = {}

        # Convert the document IDs to the document ID class.
        # This is required as the rest of TinyDB expects the document IDs
        # to be an instance of ``self.document_id_class`` but the storage
        # might convert dict keys to strings.
        table = {
            self.document_id_class(doc_id): doc
            for doc_id, doc in raw_table.items()
        }

        old_snapshots = self._index_snapshots(table)

        # Perform the table update operation
        updater(table)

        self._update_indexes(table, old_snapshots)

        # Convert the document IDs back to strings.
        # This is required as some storages (most notably the JSON file format)
        # don't support IDs other than strings.
        tables[self.name] = {
            str(doc_id): doc
            for doc_id, doc in table.items()
        }
        self._write_index_metadata(tables, table)

        # Write the newly updated data back to the storage
        self._storage.write(tables)

        # Clear the query cache, as the table contents have changed
        self.clear_cache()

    @staticmethod
    def _normalize_index_fields(
            fields: Union[str, Iterable[str]]) -> tuple[str, ...]:
        if isinstance(fields, str):
            return (fields,)
        normalized = tuple(fields)
        if not normalized or any(not isinstance(field, str) or not field
                                 for field in normalized):
            raise ValueError('Index fields must be non-empty strings')
        return normalized

    def _index_doc_id(self, doc_id: Any) -> int:
        return self.document_id_class(doc_id)

    def _read_index_metadata(self) -> dict[str, Any]:
        return self._index_metadata_from_tables(self._storage.read() or {})

    def _index_metadata_from_tables(
            self, tables: Mapping[str, Any]) -> dict[str, Any]:
        root = tables.get('__tinydb_indexes__', {})
        if not isinstance(root, Mapping):
            return {}
        metadata = root.get(self.name)
        if not isinstance(metadata, Mapping):
            return {}
        index_entries = metadata.get('indexes')
        if not isinstance(index_entries, Mapping):
            return {}
        result = {}
        for name, entry in index_entries.items():
            if isinstance(name, str) and isinstance(entry, Mapping):
                result[name] = {
                    'fields': entry.get('fields'),
                    'revision': metadata.get('revision'),
                    'checksum': metadata.get('checksum'),
                }
        return result

    def _index_snapshots(
            self, table: Mapping[int, Mapping]) -> dict[int, dict[str, tuple]]:
        snapshots: dict[int, dict[str, tuple]] = {}
        for raw_id, document in table.items():
            doc_id = self._index_doc_id(raw_id)
            document_snapshots = {}
            for name, index in self._indexes.items():
                values = []
                for field in index.fields:
                    try:
                        value = document[field]
                    except (KeyError, TypeError):
                        value = _missing
                    try:
                        values.append(copy.deepcopy(value))
                    except (TypeError, ValueError, RecursionError):
                        values.append(value)
                document_snapshots[name] = tuple(values)
            snapshots[doc_id] = document_snapshots
        return snapshots

    @staticmethod
    def _index_values(document: Mapping, index: CompositeIndex) -> tuple[Any, ...]:
        values = []
        for field in index.fields:
            try:
                values.append(document[field])
            except (KeyError, TypeError):
                values.append(_missing)
        return tuple(values)

    @staticmethod
    def _index_tuple_changed(old: tuple[Any, ...], new: tuple[Any, ...]) -> bool:
        if old is new:
            return False
        if len(old) != len(new):
            return True
        for old_value, new_value in zip(old, new):
            if old_value is _missing != (new_value is _missing):
                return True
            if old_value is _missing:
                continue
            try:
                different = old_value != new_value
            except TypeError:
                different = True
            if different:
                return True
        return False

    def _update_indexes(
            self,
            table: Mapping[int, Mapping],
            old_snapshots: Mapping[int, Mapping[str, tuple]]) -> None:
        if not self._indexes:
            return

        affected_ids = {
            self._index_doc_id(doc_id)
            for doc_id in set(old_snapshots) | set(table)
        }
        for doc_id in affected_ids:
            for name, index in self._indexes.items():
                old_values = old_snapshots.get(
                    doc_id, {}
                ).get(name, (_missing,) * len(index.fields))
                new_document = table.get(doc_id)
                new_values = (
                    self._index_values(new_document, index)
                    if new_document is not None
                    else (_missing,) * len(index.fields)
                )
                if self._index_tuple_changed(old_values, new_values):
                    index.remove(doc_id, old_values)
                    index.add(doc_id, new_values)

        self._index_revision += 1
        self._index_metadata_dirty = True

    def _index_checksum(self, table: Mapping[int, Mapping]) -> str:
        payload = []
        for raw_id in sorted(table, key=int):
            doc_id = self._index_doc_id(raw_id)
            document = table[raw_id]
            for name, index in self._indexes.items():
                values = self._index_values(document, index)
                payload.append({
                    'doc_id': doc_id,
                    'index': name,
                    'values': [
                        None if value is _missing else repr(value)
                        for value in values
                    ],
                    'missing': [value is _missing for value in values],
                })
        encoded = json.dumps(
            {'revision': self._index_revision, 'payload': payload},
            ensure_ascii=False, separators=(',', ':'), sort_keys=True,
            default=repr
        )
        return hashlib.sha256(encoded.encode('utf-8')).hexdigest()

    def _write_index_metadata(
            self, tables: dict[str, Any], table: Mapping[int, Mapping]) -> None:
        root = tables.setdefault('__tinydb_indexes__', {})
        if not isinstance(root, MutableMapping):
            root = {}
            tables['__tinydb_indexes__'] = root

        if self._indexes:
            root[self.name] = {
                'revision': self._index_revision,
                'checksum': self._index_checksum(table),
                'indexes': {
                    name: {'fields': list(index.fields)}
                    for name, index in self._indexes.items()
                },
            }
        elif isinstance(root.get(self.name), Mapping):
            del root[self.name]

        if not root:
            tables.pop('__tinydb_indexes__', None)

        self._index_metadata_dirty = False
