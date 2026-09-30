import copy

import pytest

from tinydb import TinyDB, where
from tinydb.references import (
    CASCADE,
    RESTRICT,
    SET_NULL,
    Reference,
    ReferentialIntegrityError,
)
from tinydb.storages import MemoryStorage


class RecordingStorage(MemoryStorage):
    """
    A memory storage that records every committed database state.
    """

    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, data):
        self.writes.append(copy.deepcopy(data))
        super().write(data)


@pytest.fixture
def recording_db():
    db = TinyDB(storage=RecordingStorage)
    yield db
    db.close()


@pytest.fixture
def library_db():
    """
    A database with authors and books where ``books.author_id`` references
    ``authors``.
    """
    db = TinyDB(storage=MemoryStorage)
    authors = db.table('authors')
    books = db.table('books')
    author_id = authors.insert({'name': 'Jane'})
    books.insert_multiple([
        {'title': 'One', 'author_id': author_id},
        {'title': 'Two', 'author_id': author_id},
        {'title': 'Three', 'author_id': None},
    ])
    yield db, authors, books, author_id
    db.close()


def test_restrict_rejects_delete(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    with pytest.raises(ReferentialIntegrityError):
        authors.remove(doc_ids=[author_id])

    # The failed operation did not change anything
    assert authors.get(doc_id=author_id) == {'name': 'Jane'}
    assert len(books) == 3


def test_restrict_allows_delete_of_unreferenced(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)
    other_id = authors.insert({'name': 'John'})

    authors.remove(doc_ids=[other_id])

    assert authors.get(doc_id=other_id) is None
    assert len(books) == 3


def test_set_null_on_delete(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=SET_NULL)

    authors.remove(doc_ids=[author_id])

    assert authors.get(doc_id=author_id) is None
    for book in books:
        assert book['author_id'] is None


def test_cascade_on_delete(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=CASCADE)

    removed = authors.remove(doc_ids=[author_id])

    assert removed == [author_id]
    # The two books referencing the author are gone, the one with a None
    # reference remains
    assert [book['title'] for book in books.all()] == ['Three']


def test_cascade_multiple_levels(recording_db):
    db = recording_db
    countries = db.table('countries')
    cities = db.table('cities')
    streets = db.table('streets')

    db.add_reference('cities', 'country_id', 'countries', on_delete=CASCADE)
    db.add_reference('streets', 'city_id', 'cities', on_delete=CASCADE)

    country_id = countries.insert({'name': 'Freedonia'})
    city_id = cities.insert({'name': 'Capital', 'country_id': country_id})
    streets.insert_multiple([
        {'name': 'Main', 'city_id': city_id},
        {'name': 'Side', 'city_id': city_id},
    ])
    writes_before = len(db.storage.writes)

    countries.remove(doc_ids=[country_id])

    # The whole hierarchy is gone
    assert len(countries) == 0
    assert len(cities) == 0
    assert len(streets) == 0

    # The cascade was committed with a single storage write holding the
    # final state
    assert len(db.storage.writes) == writes_before + 1
    assert db.storage.writes[-1] == db.storage.read()


def test_cascade_shared_target(recording_db):
    db = recording_db
    parents = db.table('parents')
    children = db.table('children')

    db.add_reference('children', 'parent_id', 'parents', on_delete=CASCADE)

    parent_id = parents.insert({'name': 'shared'})
    children.insert_multiple([
        {'name': 'a', 'parent_id': parent_id},
        {'name': 'b', 'parent_id': parent_id},
        {'name': 'c', 'parent_id': parent_id},
    ])

    parents.remove(doc_ids=[parent_id])

    assert len(parents) == 0
    assert len(children) == 0


def test_set_null_shared_target(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=SET_NULL)

    authors.remove(doc_ids=[author_id])

    referencing = books.search(where('author_id') == author_id)
    assert referencing == []
    assert len(books.all()) == 3


def test_cascade_cycle(recording_db):
    db = recording_db
    table_a = db.table('table_a')
    table_b = db.table('table_b')

    db.add_reference('table_a', 'b_id', 'table_b', on_delete=CASCADE)
    db.add_reference('table_b', 'a_id', 'table_a', on_delete=CASCADE)

    a_id = table_a.insert({'name': 'a'})
    b_id = table_b.insert({'name': 'b', 'a_id': a_id})
    table_a.update({'b_id': b_id}, doc_ids=[a_id])

    # Deleting one side of the cycle cascades to the other side and back;
    # the cycle is handled without looping forever
    table_a.remove(doc_ids=[a_id])

    assert len(table_a) == 0
    assert len(table_b) == 0


def test_cascade_self_reference_cycle(recording_db):
    db = recording_db
    nodes = db.table('nodes')

    db.add_reference('nodes', 'next_id', 'nodes', on_delete=CASCADE)

    first = nodes.insert({'name': 'first'})
    second = nodes.insert({'name': 'second', 'next_id': first})
    nodes.update({'next_id': second}, doc_ids=[first])

    nodes.remove(doc_ids=[first])

    assert len(nodes) == 0


def test_insert_dangling_reference_rejected(recording_db):
    db = recording_db
    books = db.table('books')
    db.add_reference('books', 'author_id', 'authors', on_delete=CASCADE)
    writes_before = len(db.storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        books.insert({'title': 'Bad', 'author_id': 999})

    # Nothing was committed
    assert len(books) == 0
    assert len(db.storage.writes) == writes_before


def test_insert_multiple_with_mutual_references(recording_db):
    db = recording_db
    nodes = db.table('nodes')
    db.add_reference('nodes', 'peer_id', 'nodes', on_delete=RESTRICT)

    # Documents inserted in the same batch may reference each other
    ids = nodes.insert_multiple([
        {'name': 'one', 'peer_id': 2},
        {'name': 'two', 'peer_id': 1},
    ])

    assert ids == [1, 2]
    assert len(nodes) == 2


def test_insert_multiple_with_dangling_reference_rejected(recording_db):
    db = recording_db
    nodes = db.table('nodes')
    db.add_reference('nodes', 'peer_id', 'nodes', on_delete=RESTRICT)
    writes_before = len(db.storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        nodes.insert_multiple([
            {'name': 'one', 'peer_id': 2},
            {'name': 'two', 'peer_id': 42},
        ])

    # The whole batch was rejected, not just the dangling document
    assert len(nodes) == 0
    assert len(db.storage.writes) == writes_before


def test_update_to_dangling_reference_rejected(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    with pytest.raises(ReferentialIntegrityError):
        books.update({'author_id': 999}, doc_ids=[1])

    assert books.get(doc_id=1)['author_id'] == author_id


def test_update_to_valid_reference_allowed(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)
    other_id = authors.insert({'name': 'John'})

    books.update({'author_id': other_id}, doc_ids=[1])

    assert books.get(doc_id=1)['author_id'] == other_id


def test_truncate_restricted(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    with pytest.raises(ReferentialIntegrityError):
        authors.truncate()

    assert len(authors) == 1
    assert len(books) == 3


def test_truncate_cascades(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=CASCADE)

    authors.truncate()

    assert len(authors) == 0
    assert [book['title'] for book in books.all()] == ['Three']


def test_drop_table_restricted(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    with pytest.raises(ReferentialIntegrityError):
        db.drop_table('authors')

    assert 'authors' in db.tables()
    assert len(authors) == 1


def test_drop_table_set_null(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=SET_NULL)

    db.drop_table('authors')

    assert 'authors' not in db.tables()
    for book in books:
        assert book['author_id'] is None


def test_drop_table_cascade(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=CASCADE)

    db.drop_table('authors')

    assert 'authors' not in db.tables()
    assert [book['title'] for book in books.all()] == ['Three']


def test_failed_delete_commits_nothing(recording_db):
    db = recording_db
    authors = db.table('authors')
    books = db.table('books')
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    author_id = authors.insert({'name': 'Jane'})
    books.insert({'title': 'One', 'author_id': author_id})
    writes_before = len(db.storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        authors.remove(doc_ids=[author_id])

    assert len(db.storage.writes) == writes_before
    assert db.storage.read() == db.storage.writes[-1]


def test_cascade_commits_single_final_state(recording_db):
    db = recording_db
    authors = db.table('authors')
    books = db.table('books')
    chapters = db.table('chapters')
    db.add_reference('books', 'author_id', 'authors', on_delete=CASCADE)
    db.add_reference('chapters', 'book_id', 'books', on_delete=CASCADE)

    author_id = authors.insert({'name': 'Jane'})
    book_id = books.insert({'title': 'One', 'author_id': author_id})
    chapters.insert_multiple([
        {'title': 'c1', 'book_id': book_id},
        {'title': 'c2', 'book_id': book_id},
    ])
    writes_before = len(db.storage.writes)

    authors.remove(doc_ids=[author_id])

    # Exactly one write for the whole multi-level cascade, and it holds
    # the final state
    assert len(db.storage.writes) == writes_before + 1
    final = db.storage.writes[-1]
    assert final == db.storage.read()
    assert final['authors'] == {}
    assert final['books'] == {}
    assert final['chapters'] == {}


def test_add_reference_rejects_violating_state():
    db = TinyDB(storage=MemoryStorage)
    books = db.table('books')
    books.insert({'title': 'Bad', 'author_id': 999})

    with pytest.raises(ReferentialIntegrityError):
        db.add_reference('books', 'author_id', 'authors')

    # The constraint was not registered
    assert db.references == ()


def test_add_reference_duplicate_is_ignored():
    db = TinyDB(storage=MemoryStorage)
    first = db.add_reference('books', 'author_id', 'authors')
    second = db.add_reference('books', 'author_id', 'authors')

    assert first == second
    assert db.references == (first,)


def test_drop_reference():
    db = TinyDB(storage=MemoryStorage)
    authors = db.table('authors')
    books = db.table('books')
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    author_id = authors.insert({'name': 'Jane'})
    books.insert({'title': 'One', 'author_id': author_id})

    db.drop_reference('books', 'author_id', 'authors')
    assert db.references == ()

    # Without the constraint the delete goes through
    authors.remove(doc_ids=[author_id])
    assert len(authors) == 0


def test_invalid_on_delete_rejected():
    with pytest.raises(ValueError):
        Reference('books', 'author_id', 'authors', on_delete='explode')


def test_reference_equality():
    assert (
        Reference('a', 'f', 'b', on_delete=CASCADE)
        == Reference('a', 'f', 'b', on_delete=CASCADE)
    )
    assert (
        Reference('a', 'f', 'b', on_delete=CASCADE)
        != Reference('a', 'f', 'b', on_delete=RESTRICT)
    )
    assert Reference('a', 'f', 'b') != 'not a reference'
    assert repr(Reference('a', 'f', 'b')).startswith('Reference(')


def test_remove_by_query_enforces_references(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    with pytest.raises(ReferentialIntegrityError):
        authors.remove(where('name') == 'Jane')

    assert len(authors) == 1


def test_none_reference_always_allowed(library_db):
    db, authors, books, author_id = library_db
    db.add_reference('books', 'author_id', 'authors', on_delete=RESTRICT)

    # Inserting a document without a reference is fine
    book_id = books.insert({'title': 'Four'})

    assert books.get(doc_id=book_id) is not None
