import pytest

from tinydb import Query, where
from tinydb.expressions import (
    DEFAULT_MAX_DEPTH,
    ExpressionError,
    OperatorRegistry,
    QueryExpression,
    default_registry,
    load_query,
)


def test_eq_roundtrip():
    expr = {'op': 'eq', 'path': 'val', 'value': 42}
    query = load_query(expr)
    direct = where('val') == 42

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'val': 42})
    assert not query({'val': 1})
    assert not query({})


def test_all_comparison_operators():
    cases = [
        ('eq', lambda q: q == 5, [5], [4, 6]),
        ('ne', lambda q: q != 5, [4], [5]),
        ('lt', lambda q: q < 5, [4], [5, 6]),
        ('le', lambda q: q <= 5, [4, 5], [6]),
        ('gt', lambda q: q > 5, [6], [4, 5]),
        ('ge', lambda q: q >= 5, [5, 6], [4]),
    ]
    for op, build, matching, non_matching in cases:
        query = load_query({'op': op, 'path': 'v', 'value': 5})
        direct = build(where('v'))

        assert query == direct, op
        assert hash(query) == hash(direct), op
        for value in matching:
            assert query({'v': value}), (op, value)
        for value in non_matching:
            assert not query({'v': value}), (op, value)
        assert not query({}), op


def test_nested_path():
    expr = {'op': 'eq', 'path': ['a', 'b', 'c'], 'value': 1}
    query = load_query(expr)
    direct = where('a')['b']['c'] == 1

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'a': {'b': {'c': 1}}})
    assert not query({'a': {'b': {'c': 2}}})
    assert not query({'a': {'b': {}}})
    assert not query({'a': 1})
    assert not query({})


def test_path_string_shorthand():
    query = load_query({'op': 'eq', 'path': 'a', 'value': 1})
    assert query == (where('a') == 1)


def test_exists():
    query = load_query({'op': 'exists', 'path': 'val'})
    direct = where('val').exists()

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'val': None})
    assert not query({'other': 1})


def test_matches_and_search():
    query = load_query({'op': 'matches', 'path': 's', 'pattern': r'^\w+$'})
    direct = where('s').matches(r'^\w+$')

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'s': 'abc'})
    assert not query({'s': 'a b'})
    assert not query({'s': 1})
    assert not query({})

    query = load_query({'op': 'search', 'path': 's', 'pattern': 'b'})
    direct = where('s').search('b')

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'s': 'abc'})
    assert not query({'s': 'ac'})


def test_matches_with_flags():
    query = load_query({'op': 'matches', 'path': 's', 'pattern': '^a+$',
                        'flags': 2})
    assert query({'s': 'AAA'})
    assert not query({'s': 'b'})


def test_one_of():
    query = load_query({'op': 'one_of', 'path': 'v', 'values': [1, 2]})
    direct = where('v').one_of([1, 2])

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'v': 1})
    assert not query({'v': 3})
    assert not query({})


def test_logical_combination():
    expr = {
        'op': 'or',
        'args': [
            {'op': 'and', 'args': [
                {'op': 'eq', 'path': 'a', 'value': 1},
                {'op': 'ne', 'path': 'b', 'value': 2},
            ]},
            {'op': 'not', 'arg': {'op': 'exists', 'path': 'c'}},
        ],
    }
    query = load_query(expr)
    direct = ((where('a') == 1) & (where('b') != 2)) | ~where('c').exists()

    assert query == direct
    assert hash(query) == hash(direct)

    docs = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 1, 'b': 2, 'c': 1},
        {'a': 2, 'c': 1},
        {'a': 2},
        {},
    ]
    for doc in docs:
        assert query(doc) == direct(doc), doc


def test_single_arg_logical():
    query = load_query({'op': 'and', 'args': [
        {'op': 'eq', 'path': 'a', 'value': 1},
    ]})
    assert query == (where('a') == 1)


def test_noop():
    query = load_query({'op': 'noop'})
    direct = Query().noop()

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({})
    assert query({'a': 1})


def test_any_with_values():
    query = load_query({'op': 'any', 'path': 'f1', 'values': [1, 2, 3]})
    direct = where('f1').any([1, 2, 3])

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'f1': [1, 2]})
    assert query({'f1': [3, 4, 5]})
    assert not query({'f1': [4, 5]})
    assert not query({'f1': 1})
    assert not query({})


def test_all_with_values():
    query = load_query({'op': 'all', 'path': 'f1', 'values': [1, 2, 3]})
    direct = where('f1').all([1, 2, 3])

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'f1': [1, 2, 3, 4, 5]})
    assert not query({'f1': [1, 2]})
    assert not query({})


def test_any_with_subquery():
    expr = {'op': 'any', 'path': 'f1',
            'where': {'op': 'eq', 'path': 'f2', 'value': 1}}
    query = load_query(expr)
    direct = where('f1').any(where('f2') == 1)

    assert query == direct
    assert hash(query) == hash(direct)

    docs = [
        {'f1': [{'f2': 1}, {'f2': 0}]},
        {'f1': [{'f2': 0}]},
        {'f1': [{'f3': 1}]},
        {'f1': []},
        {'f1': 1},
        {},
    ]
    for doc in docs:
        assert query(doc) == direct(doc), doc


def test_all_with_subquery():
    expr = {'op': 'all', 'path': 'f1',
            'where': {'op': 'eq', 'path': 'f2', 'value': 1}}
    query = load_query(expr)
    direct = where('f1').all(where('f2') == 1)

    assert query == direct
    assert hash(query) == hash(direct)

    docs = [
        {'f1': [{'f2': 1}, {'f2': 1}]},
        {'f1': [{'f2': 1}, {'f2': 0}]},
        {'f1': [{'f3': 1}]},
        {},
    ]
    for doc in docs:
        assert query(doc) == direct(doc), doc


def test_nested_arrays():
    expr = {'op': 'any', 'path': 'items',
            'where': {'op': 'any', 'path': 'tags', 'values': ['b']}}
    query = load_query(expr)
    direct = where('items').any(where('tags').any(['b']))

    assert query == direct
    assert hash(query) == hash(direct)

    docs = [
        {'items': [{'tags': ['a', 'b']}, {'tags': ['c']}]},
        {'items': [{'tags': ['a']}]},
        {'items': [{'tags': 'b'}]},
        {'items': [{}]},
        {'items': []},
        {'items': None},
        {},
    ]
    for doc in docs:
        assert query(doc) == direct(doc), doc


def test_empty_path_element_condition():
    # An empty path addresses the array element itself, allowing
    # conditions on arrays of scalars.
    expr = {'op': 'any', 'path': 'tags',
            'where': {'op': 'eq', 'path': [], 'value': 'a'}}
    query = load_query(expr)

    assert query({'tags': ['a', 'b']})
    assert not query({'tags': ['b', 'c']})
    assert not query({'tags': []})
    assert not query({})


def test_deeply_nested_arrays():
    expr = {'op': 'any', 'path': 'matrix',
            'where': {'op': 'any', 'path': [], 'values': [2]}}
    query = load_query(expr)

    assert query({'matrix': [[1, 2], [3]]})
    assert not query({'matrix': [[1], [3]]})
    assert not query({'matrix': []})
    assert not query({})


def test_parameter_binding():
    expr = {'op': 'and', 'args': [
        {'op': 'eq', 'path': 'name', 'value': {'$param': 'name'}},
        {'op': 'one_of', 'path': 'role', 'values': {'$param': 'roles'}},
    ]}
    expression = QueryExpression.parse(expr)

    assert expression.required_params == frozenset({'name', 'roles'})

    query = expression.bind({'name': 'john', 'roles': ['admin', 'staff']})
    direct = ((where('name') == 'john')
              & where('role').one_of(['admin', 'staff']))

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'name': 'john', 'role': 'admin'})
    assert not query({'name': 'john', 'role': 'user'})
    assert not query({'name': 'jane', 'role': 'admin'})


def test_parameter_binding_kwargs():
    expression = QueryExpression.parse(
        {'op': 'eq', 'path': 'a', 'value': {'$param': 'x'}})
    query = expression.bind(x=1)

    assert query == (where('a') == 1)


def test_missing_parameter_at_parse():
    expr = {'op': 'eq', 'path': 'a', 'value': {'$param': 'x'}}

    with pytest.raises(ExpressionError, match='missing parameter'):
        QueryExpression.parse(expr, params={})

    # No error when the parameter is provided
    QueryExpression.parse(expr, params={'x': 1})


def test_missing_parameter_at_bind():
    expression = QueryExpression.parse(
        {'op': 'eq', 'path': 'a', 'value': {'$param': 'x'}})

    with pytest.raises(ExpressionError, match='missing parameter'):
        expression.bind({})

    with pytest.raises(ExpressionError, match='missing parameter'):
        expression.bind()


def test_nested_parameter_reference():
    expr = {'op': 'eq', 'path': 'a',
            'value': {'outer': [{'$param': 'x'}]}}
    query = load_query(expr, {'x': [1, 2]})

    assert query({'a': {'outer': [[1, 2]]}})
    assert not query({'a': {'outer': [[1]]}})


def test_mutable_parameter_is_frozen_at_bind():
    items = [1, 2, 3]
    query = load_query({'op': 'one_of', 'path': 'v',
                        'values': {'$param': 'items'}}, {'items': items})
    hash_before = hash(query)

    # Mutating the parameter after binding must not change the query's
    # behavior or its cache hash
    items.append(4)

    assert not query({'v': 4})
    assert query({'v': 1})
    assert hash(query) == hash_before
    assert query == where('v').one_of([1, 2, 3])


def test_mutable_dict_parameter_is_frozen_at_bind():
    value = {'a': [1]}
    query = load_query({'op': 'eq', 'path': 'doc',
                        'value': {'$param': 'v'}}, {'v': value})
    hash_before = hash(query)

    value['a'].append(2)
    value['b'] = 3

    assert query({'doc': {'a': [1]}})
    assert not query({'doc': {'a': [1, 2]}})
    assert hash(query) == hash_before
    assert query == (where('doc') == {'a': [1]})


def test_json_roundtrip():
    expr = {
        'op': 'or',
        'args': [
            {'op': 'ge', 'path': ['user', 'age'], 'value': {'$param': 'min'}},
            {'op': 'any', 'path': 'tags', 'values': ['admin']},
        ],
    }
    expression = QueryExpression.parse(expr)
    restored = QueryExpression.from_json(expression.to_json())

    assert restored == expression
    assert restored.to_dict() == expression.to_dict()

    query = restored.bind({'min': 18})
    direct = ((where('user')['age'] >= 18)
              | where('tags').any(['admin']))

    assert query == direct
    assert hash(query) == hash(direct)


def test_to_dict_detached_from_expression():
    expression = QueryExpression.parse({'op': 'eq', 'path': 'a', 'value': [1]})
    dumped = expression.to_dict()
    dumped['value'].append(2)

    assert expression.bind() == (where('a') == [1])


def test_parse_detached_from_input():
    expr = {'op': 'eq', 'path': 'a', 'value': [1]}
    expression = QueryExpression.parse(expr)
    expr['value'].append(2)

    assert expression.bind() == (where('a') == [1])


def test_unknown_operator():
    with pytest.raises(ExpressionError, match='unknown operator'):
        QueryExpression.parse({'op': 'eval', 'path': 'a', 'value': 1})

    with pytest.raises(ExpressionError, match='unknown operator'):
        QueryExpression.parse({'op': '__import__'})


def test_unknown_predicate():
    with pytest.raises(ExpressionError, match='unknown predicate'):
        QueryExpression.parse({'op': 'pred', 'path': 'a',
                               'name': 'os.system'})


def test_structure_validation():
    invalid = [
        None,
        [],
        'eq',
        {},
        {'op': 1},
        {'op': 'eq'},                                # missing fields
        {'op': 'eq', 'path': 'a'},                   # missing value
        {'op': 'eq', 'path': 'a', 'value': 1,
         'extra': 2},                                # unexpected field
        {'op': 'eq', 'path': 1, 'value': 1},         # invalid path
        {'op': 'eq', 'path': ['a', 1], 'value': 1},  # invalid path segment
        {'op': 'and', 'args': []},                   # empty args
        {'op': 'and', 'args': {}},                   # args not a list
        {'op': 'not'},                               # missing arg
        {'op': 'any', 'path': 'a'},                  # no condition
        {'op': 'any', 'path': 'a', 'values': [1],
         'where': {'op': 'noop'}},                   # both conditions
        {'op': 'any', 'path': 'a', 'values': 1},     # values not a list
        {'op': 'one_of', 'path': 'a', 'values': 1},  # values not a list
        {'op': 'matches', 'path': 'a', 'pattern': 1},
        {'op': 'matches', 'path': 'a', 'pattern': 'x', 'flags': 'i'},
        {'op': 'pred', 'path': 'a', 'name': 1},
        {'op': 'eq', 'path': 'a', 'value': object()},
        {'op': 'eq', 'path': 'a', 'value': {'$param': 1}},
        {'op': 'eq', 'path': 'a', 'value': {'$param': 'x', 'other': 1}},
    ]
    for expr in invalid:
        with pytest.raises(ExpressionError):
            QueryExpression.parse(expr)


def test_depth_limit():
    expr = {'op': 'eq', 'path': 'a', 'value': 1}
    for _ in range(DEFAULT_MAX_DEPTH):
        expr = {'op': 'not', 'arg': expr}

    with pytest.raises(ExpressionError, match='maximum depth'):
        QueryExpression.parse(expr)

    # Just below the limit the same kind of expression parses fine
    expr = {'op': 'eq', 'path': 'a', 'value': 1}
    for _ in range(DEFAULT_MAX_DEPTH - 1):
        expr = {'op': 'not', 'arg': expr}

    QueryExpression.parse(expr)


def test_depth_limit_on_values():
    value = 1
    for _ in range(DEFAULT_MAX_DEPTH):
        value = [value]

    with pytest.raises(ExpressionError, match='maximum depth'):
        QueryExpression.parse({'op': 'eq', 'path': 'a', 'value': value})


def test_custom_max_depth():
    expr = {'op': 'not', 'arg': {'op': 'not', 'arg': {'op': 'noop'}}}

    QueryExpression.parse(expr, max_depth=3)
    with pytest.raises(ExpressionError, match='maximum depth'):
        QueryExpression.parse(expr, max_depth=2)


def test_custom_predicate():
    registry = OperatorRegistry.with_defaults()

    def is_even(value):
        return value % 2 == 0

    registry.register_predicate('even', is_even)

    query = QueryExpression.parse(
        {'op': 'pred', 'path': 'n', 'name': 'even'},
        registry=registry).bind()
    direct = where('n').test(is_even)

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'n': 2})
    assert not query({'n': 3})
    assert not query({})


def test_custom_predicate_with_args():
    def divisible_by(value, divisor):
        return value % divisor == 0

    registry = OperatorRegistry.with_defaults()
    registry.register_predicate('divisible_by', divisible_by)

    query = QueryExpression.parse(
        {'op': 'pred', 'path': 'n', 'name': 'divisible_by',
         'args': [{'$param': 'd'}]},
        registry=registry).bind({'d': 3})
    direct = where('n').test(divisible_by, 3)

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'n': 9})
    assert not query({'n': 10})


def test_predicate_not_leaked_between_registries():
    registry = OperatorRegistry.with_defaults()
    registry.register_predicate('mine', lambda value: True)

    expression = QueryExpression.parse(
        {'op': 'pred', 'path': 'a', 'name': 'mine'}, registry=registry)
    assert expression.bind()({'a': 1})

    # A fresh registry does not know the predicate
    with pytest.raises(ExpressionError, match='unknown predicate'):
        QueryExpression.parse({'op': 'pred', 'path': 'a', 'name': 'mine'},
                              registry=OperatorRegistry.with_defaults())


def test_custom_operator():
    registry = OperatorRegistry.with_defaults()

    def between(node, ctx):
        query = ctx.query(node['path'])
        low = ctx.resolve(node['low'])
        high = ctx.resolve(node['high'])
        return (query >= low) & (query <= high)

    registry.register_operator('between', between)

    expression = QueryExpression.parse(
        {'op': 'between', 'path': 'age',
         'low': {'$param': 'low'}, 'high': 30},
        registry=registry)
    assert expression.required_params == frozenset({'low'})

    query = expression.bind({'low': 18})
    direct = (where('age') >= 18) & (where('age') <= 30)

    assert query == direct
    assert hash(query) == hash(direct)
    assert query({'age': 20})
    assert not query({'age': 10})
    assert not query({})


def test_registry_validation():
    registry = OperatorRegistry()

    with pytest.raises(ValueError):
        registry.register_operator('', lambda node, ctx: None)

    with pytest.raises(ValueError):
        registry.register_operator('op', 'not-callable')

    with pytest.raises(ValueError):
        registry.register_predicate('', lambda value: True)

    with pytest.raises(ValueError):
        registry.register_predicate('pred', 'not-callable')


def test_default_registry_contains_builtin_operators():
    registry = default_registry()

    for op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'exists', 'matches',
               'search', 'one_of', 'any', 'all', 'and', 'or', 'not',
               'noop', 'pred'):
        assert op in registry.operators


def test_invalid_max_depth():
    with pytest.raises(ValueError):
        QueryExpression.parse({'op': 'noop'}, max_depth=0)


def test_expression_equality():
    expr = {'op': 'eq', 'path': 'a', 'value': 1}

    assert QueryExpression.parse(expr) == QueryExpression.parse(expr)
    assert QueryExpression.parse(expr) != QueryExpression.parse(
        {'op': 'eq', 'path': 'a', 'value': 2})
    assert QueryExpression.parse(expr) != 'something'
    assert repr(QueryExpression.parse(expr)).startswith('QueryExpression(')
