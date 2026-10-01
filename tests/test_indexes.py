import json

import pytest

from tinydb import Query, TinyDB, where
from tinydb.storages import JSONStorage, MemoryStorage


def make_doc(doc_id, tenant, category, score, **extra):
    return {'tenant': tenant, 'category': category, 'score': score, **extra}


def index_results_and_scan(db, query):
    assert db.uses_index(query)
    indexed = db.search(query)
    table = db.table(db.default_table_name)
    manager = table._index_manager
    original_plan_for = manager.plan_for
    manager.plan_for = lambda cond: None
    try:
        assert not table.uses_index(query)
        scanned = db.search(query)
    finally:
        manager.plan_for = original_plan_for
    assert table.uses_index(query)
    return indexed, scanned


def test_composite_index_equality_and_range(db):
    db.truncate()
    ids = db.insert_multiple([
        make_doc(1, 'a', 'x', 1),
        make_doc(2, 'a', 'x', 3),
        make_doc(3, 'a', 'x', 5),
        make_doc(4, 'a', 'y', 3),
        make_doc(5, 'b', 'x', 3),
    ])

    db.create_index('tenant', 'category', 'score')

    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') <= 3)
    )
    indexed, scanned = index_results_and_scan(db, query)
    assert [doc.doc_id for doc in indexed] == [ids[0], ids[1]]
    assert indexed == scanned

    equality_query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') == 5)
    )
    indexed, scanned = index_results_and_scan(db, equality_query)
    assert [doc.doc_id for doc in indexed] == [ids[2]]
    assert indexed == scanned

    prefix_query = (where('tenant') == 'a') & (where('category') == 'x')
    indexed, scanned = index_results_and_scan(db, prefix_query)
    assert [doc.doc_id for doc in indexed] == [ids[0], ids[1], ids[2]]
    assert indexed == scanned


def test_composite_index_is_maintained_for_updates_and_removals(db):
    db.truncate()
    ids = db.insert_multiple([
        make_doc(1, 'a', 'x', 1),
        make_doc(2, 'a', 'x', 9),
        make_doc(3, 'a', 'y', 2),
    ])
    db.create_index('tenant', 'category', 'score')
    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') < 5)
    )

    db.update({'score': 0}, doc_ids=[ids[1]])
    assert [doc.doc_id for doc in db.search(query)] == [ids[0], ids[1]]

    def move_category(document):
        document['category'] = 'y'

    db.update(move_category, doc_ids=[ids[0]])
    assert [doc.doc_id for doc in db.search(query)] == [ids[1]]

    db.update({'category': 'x'}, doc_ids=[ids[2]])
    assert [doc.doc_id for doc in db.search(query)] == [ids[1], ids[2]]

    db.remove(doc_ids=[ids[1]])
    assert [doc.doc_id for doc in db.search(query)] == [ids[2]]


def test_composite_index_is_maintained_for_update_multiple(db):
    db.truncate()
    ids = db.insert_multiple([
        {'tenant': 'a', 'category': 'x', 'score': 1, 'count': 0},
        {'tenant': 'a', 'category': 'x', 'score': 9, 'count': 0},
    ])
    db.create_index('tenant', 'category', 'score')

    db.update_multiple([
        ({'score': 2}, where('count') == 0),
        ({'score': 3}, where('score') == 2),
    ])

    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') == 3)
    )
    assert [doc.doc_id for doc in db.search(query)] == ids


def test_indexed_search_preserves_table_order(db):
    db.truncate()
    inserted = [
        {'tenant': 'a', 'category': 'x', 'score': 5},
        {'tenant': 'z', 'category': 'n', 'score': 0},
        {'tenant': 'a', 'category': 'x', 'score': 2},
        {'tenant': 'a', 'category': 'x', 'score': 9},
    ]
    db.insert_multiple(inserted)
    db.create_index('tenant', 'category', 'score')

    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') < 8)
    )
    assert [doc.doc_id for doc in db.search(query)] == [1, 3]


def test_indexed_range_evaluation_matches_and_short_circuit_order(db):
    db.truncate()
    db.insert_multiple([
        {'tenant': 'b', 'category': 'x', 'score': None},
        {'tenant': 'a', 'category': 'x', 'score': None},
        {'tenant': 'a', 'category': 'x', 'score': 2},
    ])
    db.create_index('tenant', 'category', 'score')

    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') < 5)
    )

    with pytest.raises(TypeError):
        db.search(query)

    table = db.table(db.default_table_name)
    manager = table._index_manager
    manager.plan_for = lambda cond: None
    with pytest.raises(TypeError):
        db.search(query)


def test_missing_none_and_unrelated_types_follow_scan_semantics(db):
    db.truncate()
    db.insert_multiple([
        {'tenant': 'a', 'category': 'x', 'score': 2},
        {'tenant': 'a', 'category': 'x', 'score': None},
        {'tenant': 'a', 'category': 'x'},
        {'tenant': 'a', 'category': 'x', 'score': '1'},
        {'tenant': 'a', 'category': 'x', 'score': [1, 2]},
        {'tenant': 'a', 'category': None, 'score': 2},
        {'tenant': None, 'category': 'x', 'score': 2},
        {'tenant': 'a', 'category': 'x', 'other': True},
    ])
    db.create_index('tenant', 'category', 'score')

    query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') < 5)
    )
    none_query = (
        (where('tenant') == 'a')
        & (where('category') == 'x')
        & (where('score') == None)  # noqa: E711
    )
    indexed, scanned = index_results_and_scan(db, none_query)
    assert [doc.doc_id for doc in indexed] == [2]
    assert indexed == scanned

    with pytest.raises(TypeError, match="'<' not supported"):
        db.search(query)

    db.remove(doc_ids=[2])
    with pytest.raises(TypeError, match="'<' not supported"):
        db.search(query)

    db.remove(doc_ids=[4, 5])
    indexed, scanned = index_results_and_scan(db, query)
    assert [doc.doc_id for doc in indexed] == [1]
    assert indexed == scanned

    assert len(db.all()) == 5


def test_inequivalent_queries_continue_scanning(db):
    db.truncate()
    db.insert({'tenant': 'a', 'category': 'x', 'score': 1, 'flag': True})
    db.create_index('tenant', 'category', 'score')

    queries = [
        (where('tenant') == 'a') & Query().fragment({'category': 'x'}),
        (where('tenant') == 'a') & (where('other').exists()),
        (where('tenant') == 'a') & (where('category') == 'x') & (where('score') != 1),
        (
            (where('tenant') == 'a')
            & (where('category') == 'x')
            & (where('score') == 1)
            & (where('flag') == True)  # noqa: E712
        ),
    ]

    expected_ids = [[1], [], [], [1]]
    for query, expected_id in zip(queries, expected_ids):
        assert not db.uses_index(query)
        assert [doc.doc_id for doc in db.search(query)] == expected_id


def test_reopened_valid_revision_uses_index(tmp_path):
    path = tmp_path / 'db.json'
    with TinyDB(path, storage=JSONStorage) as db:
        db.insert_multiple([
            {'tenant': 'a', 'category': 'x', 'score': 1},
            {'tenant': 'a', 'category': 'x', 'score': 2},
        ])
        db.create_index('tenant', 'category', 'score')
        revision = db.table('_default')._index_manager.revision

    with TinyDB(path, storage=JSONStorage) as db:
        db.all()
        manager = db.table('_default')._index_manager
        assert manager.revision == revision
        query = (
            (where('tenant') == 'a')
            & (where('category') == 'x')
            & (where('score') == 2)
        )
        assert db.uses_index(query)
        assert [doc.doc_id for doc in db.search(query)] == [2]
        assert len(db.all()) == 2


def test_reopened_stale_revision_rebuilds_and_does_not_return_old_result(tmp_path):
    path = tmp_path / 'db.json'
    with TinyDB(path, storage=JSONStorage) as db:
        db.insert({'tenant': 'a', 'category': 'x', 'score': 1})
        db.create_index('tenant', 'category', 'score')

    raw = json.loads(path.read_text())
    table = raw['_default']
    table['2'] = {'tenant': 'a', 'category': 'x', 'score': 2}
    path.write_text(json.dumps(raw))

    with TinyDB(path, storage=JSONStorage) as db:
        query = (
            (where('tenant') == 'a')
            & (where('category') == 'x')
            & (where('score') == 2)
        )
        assert db.uses_index(query)
        assert [doc.doc_id for doc in db.search(query)] == [2]
        assert len(db.all()) == 2
        rebuilt_revision = db.table('_default')._index_manager.revision
        db.update({'verified': True}, doc_ids=[2])

    with TinyDB(path, storage=JSONStorage) as db:
        db.all()
        assert db.table('_default')._index_manager.revision == rebuilt_revision + 1
        assert db.get(doc_id=2)['verified'] is True
