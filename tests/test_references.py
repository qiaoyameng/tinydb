import copy

import pytest

from tinydb import TinyDB, ReferentialIntegrityError
from tinydb.storages import MemoryStorage


class RecordingStorage(MemoryStorage):
    """
    A storage that records every committed write as a deep copy, so tests
    can assert that an operation commits the final state exactly once.
    """

    def __init__(self):
        super().__init__()
        self.writes = []

    def __call__(self):
        return self

    def write(self, data):
        self.writes.append(copy.deepcopy(data))
        super().write(data)


@pytest.fixture
def storage():
    return RecordingStorage()


@pytest.fixture
def db(storage):
    return TinyDB(storage=storage)


def make_blog(db, on_delete):
    db.add_reference('comments', 'post_id', 'posts', on_delete=on_delete)
    posts = db.table('posts')
    comments = db.table('comments')
    post_id = posts.insert({'title': 'hello'})
    comment_id = comments.insert({'text': 'nice', 'post_id': post_id})
    return posts, comments, post_id, comment_id


# -- restrict ---------------------------------------------------------------

def test_restrict_rejects_removal_of_referenced_document(db, storage):
    posts, comments, post_id, _ = make_blog(db, 'restrict')
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        posts.remove(doc_ids=[post_id])

    # Nothing was committed and the state is unchanged
    assert len(storage.writes) == writes_before
    assert posts.get(doc_id=post_id) is not None
    assert len(comments.all()) == 1


def test_restrict_allows_removal_of_unreferenced_document(db):
    posts, comments, post_id, _ = make_blog(db, 'restrict')
    other_id = posts.insert({'title': 'unreferenced'})

    assert posts.remove(doc_ids=[other_id]) == [other_id]
    assert posts.get(doc_id=post_id) is not None
    assert len(comments.all()) == 1


def test_restrict_rejects_truncate(db, storage):
    posts, comments, _, _ = make_blog(db, 'restrict')
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        posts.truncate()

    assert len(storage.writes) == writes_before
    assert len(posts.all()) == 1
    assert len(comments.all()) == 1


def test_restrict_rejects_drop_table(db, storage):
    posts, comments, _, _ = make_blog(db, 'restrict')
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        db.drop_table('posts')

    assert len(storage.writes) == writes_before
    assert 'posts' in db.tables()
    assert len(posts.all()) == 1
    assert len(comments.all()) == 1


# -- set_null ---------------------------------------------------------------

def test_set_null_resets_reference_on_removal(db):
    posts, comments, post_id, comment_id = make_blog(db, 'set_null')

    posts.remove(doc_ids=[post_id])

    assert posts.all() == []
    assert comments.get(doc_id=comment_id)['post_id'] is None


def test_set_null_keeps_other_references(db):
    posts, comments, post_id, _ = make_blog(db, 'set_null')
    other_id = posts.insert({'title': 'other'})
    keep_id = comments.insert({'text': 'keep', 'post_id': other_id})

    posts.remove(doc_ids=[post_id])

    assert comments.get(doc_id=keep_id)['post_id'] == other_id


def test_set_null_on_truncate(db):
    posts, comments, _, comment_id = make_blog(db, 'set_null')

    posts.truncate()

    assert posts.all() == []
    assert comments.get(doc_id=comment_id)['post_id'] is None


def test_set_null_on_drop_table(db):
    posts, comments, _, comment_id = make_blog(db, 'set_null')

    db.drop_table('posts')

    assert 'posts' not in db.tables()
    assert comments.get(doc_id=comment_id)['post_id'] is None


# -- cascade ----------------------------------------------------------------

def test_cascade_removes_referencing_documents(db):
    posts, comments, post_id, _ = make_blog(db, 'cascade')

    posts.remove(doc_ids=[post_id])

    assert posts.all() == []
    assert comments.all() == []


def test_cascade_multi_level(db, storage):
    db.add_reference('comments', 'post_id', 'posts', on_delete='cascade')
    db.add_reference('likes', 'comment_id', 'comments', on_delete='cascade')
    posts = db.table('posts')
    comments = db.table('comments')
    likes = db.table('likes')

    post_id = posts.insert({'title': 'hello'})
    comment_id = comments.insert({'text': 'nice', 'post_id': post_id})
    likes.insert({'comment_id': comment_id})
    writes_before = len(storage.writes)

    posts.remove(doc_ids=[post_id])

    # The whole chain is gone and committed with a single write
    assert posts.all() == []
    assert comments.all() == []
    assert likes.all() == []
    assert len(storage.writes) == writes_before + 1
    final = storage.writes[-1]
    assert final['posts'] == {}
    assert final['comments'] == {}
    assert final['likes'] == {}


def test_cascade_shared_target(db):
    db.add_reference('comments', 'post_id', 'posts', on_delete='cascade')
    db.add_reference('likes', 'post_id', 'posts', on_delete='cascade')
    posts = db.table('posts')
    comments = db.table('comments')
    likes = db.table('likes')

    post_id = posts.insert({'title': 'hello'})
    keep_id = posts.insert({'title': 'keep'})
    comments.insert({'text': 'a', 'post_id': post_id})
    likes.insert({'post_id': post_id})
    keep_comment_id = comments.insert({'text': 'b', 'post_id': keep_id})

    posts.remove(doc_ids=[post_id])

    # Only documents referencing the removed post are gone
    assert [doc.doc_id for doc in posts.all()] == [keep_id]
    assert [doc.doc_id for doc in comments.all()] == [keep_comment_id]
    assert likes.all() == []


def test_cascade_cycle_terminates(db, storage):
    db.add_reference('a', 'b_id', 'b', on_delete='cascade')
    db.add_reference('b', 'a_id', 'a', on_delete='cascade')
    a = db.table('a')
    b = db.table('b')

    a_id = a.insert({'name': 'a'})
    b_id = b.insert({'name': 'b', 'a_id': a_id})
    a.update({'b_id': b_id}, doc_ids=[a_id])
    writes_before = len(storage.writes)

    # Removing one side cascades around the cycle exactly once
    a.remove(doc_ids=[a_id])

    assert a.all() == []
    assert b.all() == []
    assert len(storage.writes) == writes_before + 1


def test_cascade_on_truncate(db):
    posts, comments, _, _ = make_blog(db, 'cascade')

    posts.truncate()

    assert posts.all() == []
    assert comments.all() == []


def test_cascade_on_drop_table(db):
    posts, comments, _, _ = make_blog(db, 'cascade')

    db.drop_table('posts')

    assert 'posts' not in db.tables()
    assert comments.all() == []


# -- inserts and updates ----------------------------------------------------

def test_insert_with_dangling_reference_is_rejected(db, storage):
    db.add_reference('comments', 'post_id', 'posts', on_delete='cascade')
    comments = db.table('comments')
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        comments.insert({'text': 'orphan', 'post_id': 999})

    assert len(storage.writes) == writes_before
    assert comments.all() == []


def test_insert_multiple_is_atomic(db, storage):
    db.add_reference('comments', 'post_id', 'posts', on_delete='cascade')
    posts = db.table('posts')
    comments = db.table('comments')
    post_id = posts.insert({'title': 'hello'})
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        comments.insert_multiple([
            {'text': 'ok', 'post_id': post_id},
            {'text': 'dangling', 'post_id': 999},
        ])

    # The whole batch is rejected, not just the dangling document
    assert len(storage.writes) == writes_before
    assert comments.all() == []


def test_insert_multiple_with_mutual_references_in_same_batch(db):
    db.add_reference('nodes', 'peer_id', 'nodes', on_delete='restrict')
    nodes = db.table('nodes')

    # Both documents reference each other; the batch is validated against
    # the full candidate state, so the mutual references resolve
    doc_ids = nodes.insert_multiple([
        {'name': 'first', 'peer_id': 2},
        {'name': 'second', 'peer_id': 1},
    ])

    assert doc_ids == [1, 2]
    assert len(nodes.all()) == 2


def test_insert_multiple_with_forward_reference_in_same_batch(db):
    db.add_reference('nodes', 'parent_id', 'nodes', on_delete='cascade')
    nodes = db.table('nodes')

    doc_ids = nodes.insert_multiple([
        {'name': 'root'},
        {'name': 'child', 'parent_id': 1},
    ])

    assert doc_ids == [1, 2]
    assert nodes.get(doc_id=2)['parent_id'] == 1


def test_optional_reference_allows_missing_and_null_fields(db):
    db.add_reference('comments', 'post_id', 'posts', on_delete='restrict')
    comments = db.table('comments')

    comments.insert({'text': 'no field'})
    comments.insert({'text': 'null', 'post_id': None})

    assert len(comments.all()) == 2


def test_update_to_dangling_reference_is_rejected(db, storage):
    posts, comments, _, comment_id = make_blog(db, 'restrict')
    writes_before = len(storage.writes)

    with pytest.raises(ReferentialIntegrityError):
        comments.update({'post_id': 999}, doc_ids=[comment_id])

    assert len(storage.writes) == writes_before
    assert comments.get(doc_id=comment_id)['post_id'] != 999


def test_update_to_valid_reference_is_allowed(db):
    posts, comments, _, comment_id = make_blog(db, 'restrict')
    other_id = posts.insert({'title': 'other'})

    comments.update({'post_id': other_id}, doc_ids=[comment_id])

    assert comments.get(doc_id=comment_id)['post_id'] == other_id


# -- registry ----------------------------------------------------------------

def test_reference_registry(db):
    reference = db.add_reference('comments', 'post_id', 'posts',
                                 on_delete='cascade')

    assert reference.source_table == 'comments'
    assert reference.field == 'post_id'
    assert reference.target_table == 'posts'
    assert reference.on_delete == 'cascade'
    assert db.references == (reference,)

    # Adding the same constraint twice is a no-op
    db.add_reference('comments', 'post_id', 'posts', on_delete='cascade')
    assert len(db.references) == 1

    db.remove_reference('comments', 'post_id', 'posts')
    assert db.references == ()


def test_invalid_on_delete_is_rejected(db):
    with pytest.raises(ValueError):
        db.add_reference('comments', 'post_id', 'posts',
                         on_delete='explode')


def test_removed_reference_is_no_longer_enforced(db):
    posts, comments, post_id, _ = make_blog(db, 'restrict')

    db.remove_reference('comments', 'post_id', 'posts')
    posts.remove(doc_ids=[post_id])

    assert posts.all() == []
    assert len(comments.all()) == 1


# -- commit behavior ----------------------------------------------------------

def test_single_commit_with_final_state(db, storage):
    posts, comments, post_id, _ = make_blog(db, 'cascade')
    writes_before = len(storage.writes)

    posts.remove(doc_ids=[post_id])

    # Exactly one write happened and it contains the final state
    assert len(storage.writes) == writes_before + 1
    assert storage.writes[-1] == storage.read()
    assert storage.writes[-1]['posts'] == {}
    assert storage.writes[-1]['comments'] == {}


def test_failed_operation_does_not_corrupt_memory_storage(db, storage):
    posts, comments, post_id, _ = make_blog(db, 'restrict')
    state_before = copy.deepcopy(storage.read())

    with pytest.raises(ReferentialIntegrityError):
        posts.remove(doc_ids=[post_id])

    # MemoryStorage hands out its internal state on read; a failed
    # operation must not have mutated it in place either
    assert storage.read() == state_before


def test_query_caches_of_affected_tables_are_invalidated(db):
    from tinydb import where

    posts, comments, post_id, _ = make_blog(db, 'cascade')

    # Populate the query cache of the comments table
    assert len(comments.search(where('post_id') == post_id)) == 1

    posts.remove(doc_ids=[post_id])

    # The cascade removed the comment; a cached result must not leak it
    assert comments.search(where('post_id') == post_id) == []
