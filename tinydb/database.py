"""
This module contains the main component of TinyDB: the database.
"""

import copy
from collections.abc import Iterator

from . import JSONStorage
from .references import (
    RESTRICT,
    Reference,
    enforce_references,
    validate_references,
)
from .storages import Storage
from .table import Table, Document
from .utils import with_typehint

# The table's base class. This is used to add type hinting from the Table
# class to TinyDB. Currently, this supports PyCharm, Pyright/VS Code and MyPy.
TableBase: type[Table] = with_typehint(Table)


class TinyDB(TableBase):
    """
    The main class of TinyDB.

    The ``TinyDB`` class is responsible for creating the storage class instance
    that will store this database's documents, managing the database
    tables as well as providing access to the default table.

    For table management, a simple ``dict`` is used that stores the table class
    instances accessible using their table name.

    Default table access is provided by forwarding all unknown method calls
    and property access operations to the default table by implementing
    ``__getattr__``.

    When creating a new instance, all arguments and keyword arguments (except
    for ``storage``) will be passed to the storage class that is provided. If
    no storage class is specified, :class:`~tinydb.storages.JSONStorage` will be
    used.

    .. admonition:: Customization

        For customization, the following class variables can be set:

        - ``table_class`` defines the class that is used to create tables,
        - ``default_table_name`` defines the name of the default table, and
        - ``default_storage_class`` will define the class that will be used to
          create storage instances if no other storage is passed.

        .. versionadded:: 4.0

    .. admonition:: Data Storage Model

        Data is stored using a storage class that provides persistence for a
        ``dict`` instance. This ``dict`` contains all tables and their data.
        The data is modelled like this::

            {
                'table1': {
                    0: {document...},
                    1: {document...},
                },
                'table2': {
                    ...
                }
            }

        Each entry in this ``dict`` uses the table name as its key and a
        ``dict`` of documents as its value. The document ``dict`` contains
        document IDs as keys and the documents themselves as values.

    :param storage: The class of the storage to use. Will be initialized
                    with ``args`` and ``kwargs``.
    """

    #: The class that will be used to create table instances
    #:
    #: .. versionadded:: 4.0
    table_class = Table

    #: The name of the default table
    #:
    #: .. versionadded:: 4.0
    default_table_name = '_default'

    #: The class that will be used by default to create storage instances
    #:
    #: .. versionadded:: 4.0
    default_storage_class = JSONStorage

    def __init__(self, *args, **kwargs) -> None:
        """
        Create a new instance of TinyDB.
        """

        storage = kwargs.pop('storage', self.default_storage_class)

        # Prepare the storage
        self._storage: Storage = storage(*args, **kwargs)

        self._opened = True
        self._tables: dict[str, Table] = {}
        self._references: list[Reference] = []

    def __repr__(self):

        args = [
            f'tables={list(self.tables())}',
            f'tables_count={len(self.tables())}',
            f'default_table_documents_count={self.__len__()}',
            f'all_tables_documents_count={[f"{table}={len(self.table(table))}" for table in self.tables()]}',
        ]

        return '<{} {}>'.format(type(self).__name__, ', '.join(args))

    def table(self, name: str, **kwargs) -> Table:
        """
        Get access to a specific table.

        If the table hasn't been accessed yet, a new table instance will be
        created using the :attr:`~tinydb.database.TinyDB.table_class` class.
        Otherwise, the previously created table instance will be returned.

        All further options besides the name are passed to the table class which
        by default is :class:`~tinydb.table.Table`. Check its documentation
        for further parameters you can pass.

        :param name: The name of the table.
        :param kwargs: Keyword arguments to pass to the table class constructor
        """

        if name in self._tables:
            return self._tables[name]

        table = self.table_class(self.storage, name, **kwargs)

        # Give the table access to this database's referential constraints
        table._database = self

        self._tables[name] = table

        return table

    def tables(self) -> set[str]:
        """
        Get the names of all tables in the database.

        :returns: a set of table names
        """

        # TinyDB stores data as a dict of tables like this:
        #
        #   {
        #       '_default': {
        #           0: {document...},
        #           1: {document...},
        #       },
        #       'table1': {
        #           ...
        #       }
        #   }
        #
        # To get a set of table names, we thus construct a set of this main
        # dict which returns a set of the dict keys which are the table names.
        #
        # Storage.read() may return ``None`` if the database file is empty,
        # so we need to consider this case to and return an empty set in this
        # case.

        return set(self.storage.read() or {})

    @property
    def references(self) -> tuple[Reference, ...]:
        """
        Get the referential constraints registered on this database.

        :returns: a tuple of all registered :class:`Reference` objects
        """
        return tuple(self._references)

    def add_reference(
        self,
        table: str,
        field: str,
        ref_table: str,
        on_delete: str = RESTRICT,
    ) -> Reference:
        """
        Register a referential constraint between two tables.

        Declares that ``field`` of the documents in ``table`` stores the
        ``doc_id`` of a document in ``ref_table``. From now on, every
        database write validates the complete candidate state against all
        registered references before it is committed: deleting a referenced
        document triggers the ``on_delete`` behaviour (``RESTRICT`` rejects
        the operation, ``SET_NULL`` nulls the referencing field, ``CASCADE``
        deletes the referencing document), and writes that would introduce
        a reference to a non-existing document are rejected. A field value
        of ``None`` (or a missing field) means "no reference".

        :param table: the name of the table holding the referencing field
        :param field: the document field storing the target ``doc_id``
        :param ref_table: the name of the table the field points to
        :param on_delete: one of ``'restrict'``, ``'set_null'`` or
                          ``'cascade'``
        :returns: the registered :class:`Reference`
        :raises ReferentialIntegrityError: if the current database state
            already violates the new constraint
        """
        reference = Reference(table, field, ref_table, on_delete)

        if reference in self._references:
            return reference

        # Make sure the current state satisfies the new constraint
        validate_references(
            self.storage.read() or {},
            self._references + [reference],
        )

        self._references.append(reference)

        return reference

    def drop_reference(self, table: str, field: str, ref_table: str) -> None:
        """
        Remove a previously registered referential constraint.

        :param table: the name of the table holding the referencing field
        :param field: the document field storing the target ``doc_id``
        :param ref_table: the name of the table the field points to
        """
        self._references = [
            reference for reference in self._references
            if not (
                reference.table == table
                and reference.field == field
                and reference.ref_table == ref_table
            )
        ]

    def drop_tables(self) -> None:
        """
        Drop all tables from the database. **CANNOT BE REVERSED!**
        """

        # We drop all tables from this database by writing an empty dict
        # to the storage thereby returning to the initial state with no tables.
        self.storage.write({})

        # After that we need to remember to empty the ``_tables`` dict, so we'll
        # create new table instances when a table is accessed again.
        self._tables.clear()

    def drop_table(self, name: str) -> None:
        """
        Drop a specific table from the database. **CANNOT BE REVERSED!**

        Dropping a table deletes all its documents, so the registered
        referential constraints are enforced the same way as for any other
        deletion: depending on their ``on_delete`` behaviour, references to
        the dropped table's documents either reject the operation, are set
        to ``None`` or delete the referencing documents.

        :param name: The name of the table to drop.
        """

        data = self.storage.read()

        # The database is uninitialized, there's nothing to do
        if data is None:
            return

        # The table does not exist, there's nothing to do
        if name not in data:
            return

        if self._references:
            # Validate the complete candidate state (without the dropped
            # table) on a deep copy, so a constraint violation leaves the
            # storage untouched and the final state is written exactly once
            candidate = copy.deepcopy(data)
            del candidate[name]

            # May raise ReferentialIntegrityError, aborting the drop
            enforce_references(data, candidate, self._references)

            self.storage.write(candidate)
        else:
            # Remove the table from the data dict
            del data[name]

            # Store the updated data back to the storage
            self.storage.write(data)

        # If the table is currently opened, we need to forget the table class
        # instance
        if name in self._tables:
            del self._tables[name]

    @property
    def storage(self) -> Storage:
        """
        Get the storage instance used for this TinyDB instance.

        :return: This instance's storage
        :rtype: Storage
        """
        return self._storage

    def close(self) -> None:
        """
        Close the database.

        This may be needed if the storage instance used for this database
        needs to perform cleanup operations like closing file handles.

        To ensure this method is called, the TinyDB instance can be used as a
        context manager::

            with TinyDB('data.json') as db:
                db.insert({'foo': 'bar'})

        Upon leaving this context, the ``close`` method will be called.
        """
        self._opened = False
        self.storage.close()

    def __enter__(self):
        """
        Use the database as a context manager.

        Using the database as a context manager ensures that the
        :meth:`~tinydb.database.TinyDB.close` method is called upon leaving
        the context.

        :return: The current instance
        """
        return self

    def __exit__(self, *args):
        """
        Close the storage instance when leaving a context.
        """
        if self._opened:
            self.close()

    def __getattr__(self, name):
        """
        Forward all unknown attribute calls to the default table instance.
        """
        return getattr(self.table(self.default_table_name), name)

    # Here we forward magic methods to the default table instance. These are
    # not handled by __getattr__ so we need to forward them manually here

    def __len__(self):
        """
        Get the total number of documents in the default table.

        >>> db = TinyDB('db.json')
        >>> len(db)
        0
        """
        return len(self.table(self.default_table_name))

    def __iter__(self) -> Iterator[Document]:
        """
        Return an iterator for the default table's documents.
        """
        return iter(self.table(self.default_table_name))
