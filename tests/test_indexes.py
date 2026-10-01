import json

import pytest

from tinydb import TinyDB, where
from tinydb.indexes import CompositeIndex
from tinydb.storages import JSONStorage, MemoryStorage


RANGE_METHODS = {
    '<': '__lt__',
    '<=': '__le__',
    '>': '__gt__',
    '>=': '__ge__',
}


def test_maintain_composite_index_for_insert_update_and_remove(db):
    table = db.table('indexed')
    table.insert_multiple([
        {'category': 'a', 'group': 'x', 'score': 1},
        {'category': 'a', 'group': 'x', 'score': 5},
        {'category': 'a', 'group': 'y', 'score': 2},
        {'category': 'b', 'group': 'x', 'score': 3},
    ])
    table.add_index(('category', 'group', 'score'), name='main')

    query = (where('category') == 'a') & (where('group') == 'x') & \
            (where('score') >= 2)
    assert table._index_plan(query).index is table.indexes()['main']
    assert [doc['score'] for doc in table.search(query)] == [5]

    table.update({'score': 0}, doc_ids=[1])
    table.update({'category': 'b'}, doc_ids=[2])
    assert table._search_scan(query) == table.search(query)
    assert table.search(query) == []

    table.remove(doc_ids=[4])
    assert table._search_scan(query) == table.search(query)


def test_index_covers_full_equality_and_prefixes(db):
    table = db.table('indexed')
    table.insert_multiple([
        {'category': 'a', 'group': 'x', 'score': 1},
        {'category': 'a', 'group': 'x', 'score': 2},
        {'category': 'a', 'group': 'y', 'score': 3},
        {'category': 'b', 'group': 'x', 'score': 4},
    ])
    table.add_index(('category', 'group', 'score'), name='main')

    queries = [
        (where('category') == 'a') & (where('group') == 'x'),
        where('category') == 'a',
        (where('category') == 'a') & (where('group') == 'x') &
        (where('score') == 2),
    ]

    for query in queries:
        assert table._index_plan(query) is not None
        assert table.search(query) == table._search_scan(query)


def test_only_equivalent_query_combinations_use_index(db):
    table = db.table('indexed')
    table.insert({'category': 'a', 'group': 'x', 'score': 1, 'extra': 2})
    table.add_index(('category', 'group', 'score'), name='main')

    supported = [
        (where('category') == 'a') & (where('group') == 'x') &
        (where('score') > 0),
        (where('group') == 'x') & (where('score') > 0) &
        (where('category') == 'a'),
        where('category') == 'a',
    ]
    unsupported = [
        (where('category') == 'a') & (where('score') > 0),
        (where('group') == 'x') & (where('score') > 0),
        (where('category') == 'a') | (where('group') == 'x'),
        ~(where('category') == 'a'),
        (where('category') == 'a') & (where('group') == 'x') &
        (where('score').exists()),
        (where('category') == 'a') & (where('group') == 'x') &
        (where('score') > 0) & (where('extra') == 2),
        where('category').exists(),
    ]

    for query in supported:
        assert table._index_plan(query) is not None

    for query in unsupported:
        assert table._index_plan(query) is None
        assert table.search(query) == table._search_scan(query)


def test_missing_none_and_heterogeneous_types_match_scan_semantics(db):
    table = db.table('indexed')
    documents = [
        {'category': 'a', 'group': 'x', 'score': 1},
        {'category': 'a', 'group': 'x', 'score': 4},
        {'category': 'a', 'group': 'x', 'score': '2'},
        {'category': 'a', 'group': 'x', 'score': None},
        {'category': 'a', 'group': 'x'},
        {'category': 'a', 'group': 'x', 'score': True},
    ]
    table.insert_multiple(documents)
    table.add_index(('category', 'group', 'score'), name='main')

    equality = (where('category') == 'a') & (where('group') == 'x') & \
               (where('score') == None)
    assert table.search(equality) == table._search_scan(equality)
    assert table.search(equality) == [documents[3]]

    assert table.search(where('score').exists()) == \
        table._search_scan(where('score').exists())

    for operator, method in RANGE_METHODS.items():
        query = (
            (where('category') == 'a') &
            (where('group') == 'x') &
            (getattr(where('score'), method)(2))
        )
        assert table._index_plan(query) is not None
        with pytest.raises(TypeError):
            table._search_scan(query)
        with pytest.raises(TypeError):
            table.search(query)

    single_none = where('score') == None
    assert table._index_plan(single_none) is None
    assert table.search(single_none) == table._search_scan(single_none)


def test_index_preserves_storage_document_order(db):
    table = db.table('indexed')
    table.insert_multiple([
        {'category': 'a', 'group': 'x', 'score': 9},
        {'category': 'a', 'group': 'x', 'score': 1},
        {'category': 'a', 'group': 'x', 'score': 5},
        {'category': 'a', 'group': 'x', 'score': 2},
    ])
    table.add_index(('category', 'group', 'score'))

    query = (where('category') == 'a') & (where('group') == 'x') & \
            (where('score') > 0)
    assert [doc.doc_id for doc in table.search(query)] == [1, 2, 3, 4]
    assert table.search(query) == table._search_scan(query)


def test_unhashable_values_and_fallback_to_scan():
    class Incomparable:
        def __lt__(self, other):
            raise TypeError('unsupported ordering')

        def __gt__(self, other):
            raise TypeError('unsupported ordering')

        def __le__(self, other):
            raise TypeError('unsupported ordering')

        def __ge__(self, other):
            raise TypeError('unsupported ordering')

    db = TinyDB(storage=MemoryStorage)
    table = db.table('indexed')
    table.insert_multiple([
        {'category': 'a', 'group': 'x', 'score': [1]},
        {'category': 'a', 'group': 'x', 'score': [2]},
        {'category': 'a', 'group': 'x', 'score': Incomparable()},
    ])
    table.add_index(('category', 'group', 'score'))

    equality = (where('category') == 'a') & (where('group') == 'x') & \
               (where('score') == [2])
    assert table.search(equality) == table._search_scan(equality)

    range_query = (where('category') == 'a') & (where('group') == 'x') & \
                  (where('score') < [3])
    assert table._index_plan(range_query) is not None
    with pytest.raises(TypeError):
        table._search_scan(range_query)
    with pytest.raises(TypeError):
        table.search(range_query)
    db.close()


def test_equality_follows_python_value_semantics(db):
    table = db.table('indexed')
    table.insert_multiple([
        {'category': 'a', 'group': 'x', 'score': [1, 2]},
        {'category': 'a', 'group': 'x', 'score': 1},
        {'category': 'a', 'group': 'x', 'score': True},
    ])
    table.add_index(('category', 'group', 'score'))

    list_query = (
        (where('category') == 'a') &
        (where('group') == 'x') &
        (where('score') == [1, 2])
    )
    assert table.search(list_query) == table._search_scan(list_query)
    assert len(table.search(list_query)) == 1

    bool_query = (
        (where('category') == 'a') &
        (where('group') == 'x') &
        (where('score') == 1)
    )
    assert table.search(bool_query) == table._search_scan(bool_query)
    assert len(table.search(bool_query)) == 2


def test_reopen_validates_revision_and_rebuilds_stale_index(tmp_path):
    path = tmp_path / 'db.json'

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        table.insert_multiple([
            {'category': 'a', 'group': 'x', 'score': 1},
            {'category': 'a', 'group': 'x', 'score': 4},
        ])
        table.add_index(('category', 'group', 'score'))

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        assert table.indexes()
        query = (where('category') == 'a') & (where('group') == 'x') & \
                (where('score') > 2)
        assert [doc['score'] for doc in table.search(query)] == [4]

    raw = json.loads(path.read_text())
    raw['indexed']['1']['score'] = 9
    path.write_text(json.dumps(raw))

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        query = (where('category') == 'a') & (where('group') == 'x') & \
                (where('score') > 5)
        assert [doc['score'] for doc in table.search(query)] == [9]
        assert table.search(query) == table._search_scan(query)

    repaired = json.loads(path.read_text())
    metadata = repaired['__tinydb_indexes__']['indexed']
    assert metadata['revision'] == 2


def test_reopen_rebuilds_when_revision_metadata_is_stale(tmp_path):
    path = tmp_path / 'db.json'

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        table.insert({'category': 'a', 'score': 1})
        table.add_index(('category', 'score'))

    raw = json.loads(path.read_text())
    raw['__tinydb_indexes__']['indexed']['revision'] = 999
    path.write_text(json.dumps(raw))

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        query = where('category') == 'a'
        assert table.search(query) == table._search_scan(query)

    repaired = json.loads(path.read_text())
    assert repaired['__tinydb_indexes__']['indexed']['revision'] == 1000


def test_drop_table_removes_index_metadata_and_state(tmp_path):
    path = tmp_path / 'db.json'

    with TinyDB(path, storage=JSONStorage) as db:
        table = db.table('indexed')
        table.insert({'category': 'a', 'score': 1})
        table.add_index(('category', 'score'))
        db.drop_table('indexed')

        assert '__tinydb_indexes__' not in db.storage.read()
        assert table.indexes() == {}


def test_default_table_constructor_indexes(tmp_path):
    path = tmp_path / 'db.json'
    indexes = {'category_score': ('category', 'score')}

    with TinyDB(path, storage=JSONStorage, indexes=indexes) as db:
        db.insert_multiple([
            {'category': 'a', 'score': 1},
            {'category': 'a', 'score': 3},
        ])

    with TinyDB(path, storage=JSONStorage, indexes=indexes) as db:
        assert db._index_plan(where('category') == 'a') is not None
        assert [doc['score'] for doc in db.search(
            (where('category') == 'a') & (where('score') >= 2)
        )] == [3]


def test_indexes_work_with_memory_storage():
    db = TinyDB(
        storage=MemoryStorage,
        indexes={'category_score': ('category', 'score')},
    )
    db.insert({'category': 'a', 'score': 1})
    query = where('category') == 'a'
    assert db._index_plan(query) is not None
    assert db.search(query) == db._search_scan(query)
    db.close()


def test_single_field_range_index_preserves_scan_behavior():
    db = TinyDB(
        storage=MemoryStorage,
        indexes={'score_index': ('score',)},
    )
    db.insert_multiple([{'score': 3}, {'score': 1}, {'other': True}])

    range_query = where('score') > 1
    assert db._index_plan(range_query) is not None
    assert [doc['score'] for doc in db.search(range_query)] == [3]

    none_range = where('score') < None
    assert db._index_plan(none_range) is not None
    with pytest.raises(TypeError):
        db._search_scan(none_range)
    with pytest.raises(TypeError):
        db.search(none_range)
    db.close()


def test_composite_index_rejects_invalid_fields():
    db = TinyDB(storage=MemoryStorage)
    try:
        with pytest.raises(ValueError):
            CompositeIndex(())
        with pytest.raises(ValueError):
            CompositeIndex(('a', 'a'))
    finally:
        db.close()
