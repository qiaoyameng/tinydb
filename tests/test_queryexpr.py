import json
import re

import pytest

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage
from tinydb.queryexpr import (
    compile_query_expression,
    PredicateRegistry,
    InvalidQueryExpressionError,
    UnknownOperatorError,
    MissingParameterError,
    MAX_EXPRESSION_DEPTH,
)


def cmp_ops():
    return {
        'eq': lambda field, value: field == value,
        'ne': lambda field, value: field != value,
        'lt': lambda field, value: field < value,
        'le': lambda field, value: field <= value,
        'gt': lambda field, value: field > value,
        'ge': lambda field, value: field >= value,
    }


@pytest.mark.parametrize('op', ['eq', 'ne', 'lt', 'le', 'gt', 'ge'])
def test_comparison_parity(op):
    expr = {'op': op, 'path': ['a'], 'value': {'$param': 'v'}}
    compiled = compile_query_expression(expr, {'v': 5})
    direct = cmp_ops()[op](Query().a, 5)

    docs = [{'a': 4}, {'a': 5}, {'a': 6}, {}]
    for doc in docs:
        assert compiled(doc) is direct(doc)
    assert hash(compiled) == hash(direct)
    assert compiled == direct


def test_nested_path_parity():
    expr = {'op': 'eq', 'path': ['a', 'b'], 'value': {'$param': 'v'}}
    compiled = compile_query_expression(expr, {'v': 1})
    direct = Query().a.b == 1

    docs = [
        {'a': {'b': 1}},
        {'a': {'b': 2}},
        {'a': 1},
        {'a': [{'b': 1}]},
        {},
    ]
    for doc in docs:
        assert compiled(doc) is direct(doc)


def test_inline_value_and_logic():
    expr = {
        'op': 'or',
        'left': {
            'op': 'and',
            'left': {'op': 'eq', 'path': ['f1'], 'value': 5},
            'right': {'op': 'ne', 'path': ['f2'], 'value': 2},
        },
        'right': {'op': 'not', 'cond': {'op': 'exists', 'path': ['f2']}},
    }
    compiled = compile_query_expression(expr)
    direct = ((Query().f1 == 5) & (Query().f2 != 2)) | ~Query().f2.exists()

    docs = [
        {'f1': 5, 'f2': 3},
        {'f1': 5, 'f2': 2},
        {'f1': 4},
        {},
    ]
    for doc in docs:
        assert compiled(doc) is direct(doc)
    assert compiled == direct


def test_exists_parity():
    compiled = compile_query_expression(
        {'op': 'exists', 'path': ['a', 'b']}
    )
    direct = Query().a.b.exists()
    for doc in [{'a': {'b': 0}}, {'a': {}}, {}]:
        assert compiled(doc) is direct(doc)
    assert compiled == direct


@pytest.mark.parametrize('op,method', [('matches', 'matches'), ('search', 'search')])
def test_regex_parity(op, method):
    compiled = compile_query_expression(
        {'op': op, 'path': ['s'], 'pattern': r'^\w+$', 'flags': re.IGNORECASE}
    )
    direct = getattr(Query().s, method)(r'^\w+$', re.IGNORECASE)
    for doc in [{'s': 'AbC'}, {'s': 'a b'}, {'s': 1}]:
        assert compiled(doc) is direct(doc)
    assert compiled == direct


def test_one_of_parity():
    values = ['x', 'y']
    compiled = compile_query_expression(
        {'op': 'one_of', 'path': ['v'], 'values': {'$param': 'values'}},
        {'values': values},
    )
    direct = Query().v.one_of(['x', 'y'])
    for doc in [{'v': 'x'}, {'v': 'z'}, {}]:
        assert compiled(doc) is direct(doc)
    assert compiled == direct


def test_any_all_expression_cond_parity():
    for op, method in [('any', 'any'), ('all', 'all')]:
        expr = {
            'op': op,
            'path': ['items'],
            'cond': {'op': 'eq', 'path': ['f2'], 'value': 1},
        }
        compiled = compile_query_expression(expr)
        direct = getattr(Query().items, method)(Query().f2 == 1)
        docs = [
            {'items': [{'f2': 1}, {'f2': 0}]},
            {'items': [{'f2': 1}, {'f2': 1}]},
            {'items': []},
            {'items': 'abc'},
            {},
        ]
        for doc in docs:
            assert compiled(doc) is direct(doc), (op, doc)
        assert compiled == direct


def test_any_all_list_cond_parity():
    expr_any = {'op': 'any', 'path': ['f1'], 'cond': {'$param': 'cond'}}
    compiled = compile_query_expression(
        expr_any, {'cond': [1, 2, 3]}
    )
    direct = Query().f1.any([1, 2, 3])
    for doc in [{'f1': [1, 2]}, {'f1': [3, 4, 5]}, {'f1': []}, {}]:
        assert compiled(doc) is direct(doc)
    assert compiled == direct

    compiled_all = compile_query_expression(
        {'op': 'all', 'path': ['f1'], 'cond': [1, 2]},
    )
    direct_all = Query().f1.all([1, 2])
    for doc in [{'f1': [1, 2, 3]}, {'f1': [1, 3]}, {'f1': []}, {}]:
        assert compiled_all(doc) is direct_all(doc)


def test_nested_arrays_roundtrip():
    expr = {
        'op': 'any',
        'path': ['groups'],
        'cond': {
            'op': 'all',
            'path': ['members'],
            'cond': {'op': 'ge', 'path': ['age'], 'value': 18},
        },
    }
    compiled = compile_query_expression(expr)
    restored = compile_query_expression(
        json.loads(json.dumps(expr))
    )
    docs = [
        {'groups': [{'members': [{'age': 18}, {'age': 21}]}]},
        {'groups': [{'members': [{'age': 12}]}]},
        {'groups': []},
        {},
    ]
    for doc in docs:
        assert compiled(doc) is restored(doc)
    assert compiled == restored


def test_custom_predicate_from_registry():
    registry = PredicateRegistry()

    @registry.register
    def divisible(value, divisor):
        return value % divisor == 0

    expr = {
        'op': 'pred',
        'path': ['n'],
        'name': 'divisible',
        'args': [{'$param': 'divisor'}],
    }
    compiled = compile_query_expression(expr, {'divisor': 3}, predicates=registry)
    direct = Query().n.test(divisible, 3)
    for doc in [{'n': 9}, {'n': 10}, {}]:
        assert compiled(doc) is direct(doc)
    assert compiled == direct


def test_unknown_predicate_rejected():
    expr = {'op': 'pred', 'path': ['n'], 'name': 'nope'}
    with pytest.raises(UnknownOperatorError):
        compile_query_expression(expr, predicates=PredicateRegistry())

    with pytest.raises(UnknownOperatorError):
        compile_query_expression(expr)


def test_predicate_registry_duplicate_name():
    registry = PredicateRegistry()
    registry.register(lambda v: True, name='p')
    with pytest.raises(ValueError):
        registry.register(lambda v: False, name='p')


def test_missing_parameter():
    expr = {'op': 'eq', 'path': ['a'], 'value': {'$param': 'v'}}
    with pytest.raises(MissingParameterError):
        compile_query_expression(expr)
    with pytest.raises(MissingParameterError):
        compile_query_expression(expr, {'other': 1})


def test_mutation_after_binding_does_not_change_query():
    values = [1, 2]
    expr = {'op': 'one_of', 'path': ['v'], 'values': {'$param': 'values'}}
    compiled = compile_query_expression(expr, {'values': values})

    assert compiled({'v': 1})
    values.append(99)
    assert not compiled({'v': 99})

    # The query cache key must stay stable too.
    before = hash(compiled)
    values.extend(range(100))
    assert hash(compiled) == before


def test_deep_mutation_after_binding():
    params = {'bound': {'tags': ['a']}}
    expr = {
        'op': 'eq',
        'path': ['v'],
        'value': {'$param': 'bound'},
    }
    compiled = compile_query_expression(expr, params)
    assert compiled({'v': {'tags': ['a']}})
    params['bound']['tags'].append('b')
    assert not compiled({'v': {'tags': ['a', 'b']}})
    assert compiled({'v': {'tags': ['a']}})


def test_unknown_operator_rejected():
    with pytest.raises(UnknownOperatorError):
        compile_query_expression(
            {'op': '__import__', 'path': ['a'], 'value': 1}
        )
    with pytest.raises(UnknownOperatorError):
        compile_query_expression({'op': 'eval', 'cond': {}})


@pytest.mark.parametrize(
    'bad',
    [
        [],
        'eq',
        42,
        None,
        {'path': ['a'], 'value': 1},
        {'op': 1},
        {'op': 'eq', 'path': 'a', 'value': 1},
        {'op': 'eq', 'path': [], 'value': 1},
        {'op': 'eq', 'path': [1], 'value': 1},
        {'op': 'eq', 'path': ['a']},
        {'op': 'eq', 'path': ['a'], 'value': 1, 'extra': 2},
        {'op': 'and', 'left': {'op': 'exists', 'path': ['a']}},
        {'op': 'and', 'left': [], 'right': {'op': 'exists', 'path': ['a']}},
        {'op': 'not'},
        {'op': 'not', 'cond': []},
        {'op': 'exists', 'path': ['a'], 'bogus': True},
        {'op': 'matches', 'path': ['s'], 'pattern': 5},
        {'op': 'matches', 'path': ['s'], 'pattern': 'x', 'flags': 'i'},
        {'op': 'search', 'path': ['s']},
        {'op': 'one_of', 'path': ['v'], 'values': 'x'},
        {'op': 'one_of', 'path': ['v'], 'values': {'$param': 'p'}},
        {'op': 'any', 'path': ['v'], 'cond': 5},
        {'op': 'any', 'path': ['v'], 'cond': {'$param': 'p'}},
        {'op': 'pred', 'path': ['v'], 'name': 1},
        {'op': 'pred', 'path': ['v'], 'name': 'p', 'args': {}},
    ],
)
def test_invalid_structures(bad):
    params = {'p': 5}
    with pytest.raises(InvalidQueryExpressionError):
        compile_query_expression(bad, params, predicates=PredicateRegistry())


def test_depth_limit():
    node = {'op': 'exists', 'path': ['a']}
    for _ in range(5):
        node = {'op': 'not', 'cond': node}

    assert compile_query_expression(node, max_depth=10)
    with pytest.raises(InvalidQueryExpressionError):
        compile_query_expression(node, max_depth=3)

    default_limit_node = {'op': 'exists', 'path': ['a']}
    for _ in range(MAX_EXPRESSION_DEPTH + 1):
        default_limit_node = {'op': 'not', 'cond': default_limit_node}
    with pytest.raises(InvalidQueryExpressionError):
        compile_query_expression(default_limit_node)


def test_cycle_rejected():
    node = {'op': 'and', 'left': {'op': 'exists', 'path': ['a']}}
    node['right'] = node
    with pytest.raises(InvalidQueryExpressionError):
        compile_query_expression(node)


def test_json_roundtrip_with_table_cache():
    expr = {
        'op': 'and',
        'left': {'op': 'ge', 'path': ['age'], 'value': {'$param': 'min_age'}},
        'right': {
            'op': 'any',
            'path': ['tags'],
            'cond': {'$param': 'tag_values'},
        },
    }
    params = {'min_age': 18, 'tag_values': ['admin', 'user']}

    restored_expr = json.loads(json.dumps(expr))
    query = compile_query_expression(restored_expr, params)
    direct = (Query().age >= 18) & Query().tags.any(['admin', 'user'])
    assert query == direct

    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple([
        {'age': 20, 'tags': ['user']},
        {'age': 12, 'tags': ['admin']},
        {'age': 30, 'tags': ['guest']},
    ])
    assert db.search(query) == [{'age': 20, 'tags': ['user']}]

    # Repeated identical compiled queries reuse the cache.
    same = compile_query_expression(json.loads(json.dumps(expr)), params)
    table = db.table('_default')
    table._query_cache.clear()
    table.search(same)
    assert same in table._query_cache
    results = table.search(same)
    assert results == [{'age': 20, 'tags': ['user']}]

    # Mutating bindings after compile must not alter cached semantics.
    params['tag_values'].append('guest')
    params['min_age'] = 0
    assert db.search(query) == [{'age': 20, 'tags': ['user']}]
