"""
Serializable query expressions.

TinyDB queries built via :class:`~tinydb.queries.Query` may contain
arbitrary Python callables, so they can neither be serialized to JSON nor
safely restored without ``pickle``. This module provides a small,
explicitly bounded expression language that covers the common query
operations and can be stored as plain JSON:

>>> from tinydb.queryexpr import compile_query_expression
>>> expr = {
...     'op': 'and',
...     'left': {'op': 'eq', 'path': ['name'], 'value': {'$param': 'name'}},
...     'right': {'op': 'ge', 'path': ['age'], 'value': {'$param': 'min_age'}},
... }
>>> query = compile_query_expression(expr, {'name': 'John', 'min_age': 21})
>>> query({'name': 'John', 'age': 42})
True

Expressions are compiled into regular
:class:`~tinydb.queries.QueryInstance` objects, so evaluating a restored
expression behaves exactly like constructing the equivalent ``Query``
directly -- including how missing fields and nested arrays are handled.

Only a fixed set of operators is available, and custom predicates must be
registered explicitly in a :class:`PredicateRegistry`. No expression can
trigger imports or evaluate arbitrary code.
"""

import copy
from collections.abc import Mapping, Callable
from typing import Any, Optional, Union

from .queries import Query, QueryInstance

__all__ = (
    'compile_query_expression',
    'PredicateRegistry',
    'QueryExpressionError',
    'InvalidQueryExpressionError',
    'UnknownOperatorError',
    'MissingParameterError',
    'MAX_EXPRESSION_DEPTH',
)

#: Maximum allowed nesting depth of an expression.
MAX_EXPRESSION_DEPTH = 32

#: Operators that compare the value found at ``path`` against ``value``.
_COMPARISON_OPS = frozenset(('eq', 'ne', 'lt', 'le', 'gt', 'ge'))

#: Operators that take a single nested expression.
_UNARY_OPS = frozenset(('not',))

#: Operators that combine two nested expressions.
_BINARY_OPS = frozenset(('and', 'or'))

#: Operators that evaluate nested expressions against array elements.
_ARRAY_OPS = frozenset(('any', 'all'))

#: All operators that are built into the expression language.
_BUILTIN_OPS = (
    _COMPARISON_OPS
    | _UNARY_OPS
    | _BINARY_OPS
    | _ARRAY_OPS
    | frozenset(('exists', 'matches', 'search', 'one_of', 'pred'))
)

_PARAM_KEY = '$param'


class QueryExpressionError(ValueError):
    """Base class for all query expression related errors."""


class InvalidQueryExpressionError(QueryExpressionError):
    """Raised when an expression has an invalid structure."""


class UnknownOperatorError(QueryExpressionError):
    """Raised when an expression uses an operator that is not registered."""


class MissingParameterError(QueryExpressionError):
    """Raised when an expression references a parameter that was not bound."""


class PredicateRegistry:
    """
    An explicit registry of custom predicates usable from expressions.

    Predicates are plain callables that receive the value found at the
    expression's path, followed by any bound arguments::

        registry = PredicateRegistry()

        @registry.register
        def is_even(value):
            return value % 2 == 0

        expr = {'op': 'pred', 'path': ['n'], 'name': 'is_even'}

    Only predicates registered on the registry passed to
    :func:`compile_query_expression` can be referenced; expressions cannot
    import or look up arbitrary callables.
    """

    def __init__(self) -> None:
        self._predicates: dict[str, Callable[..., bool]] = {}

    def register(
        self,
        func: Optional[Callable[..., bool]] = None,
        *,
        name: Optional[str] = None,
    ) -> Any:
        """
        Register a predicate.

        Can be used directly (``registry.register(func)``) or as a
        decorator (``@registry.register`` or
        ``@registry.register(name='...')``).
        """

        def do_register(fn: Callable[..., bool]) -> Callable[..., bool]:
            predicate_name = name or fn.__name__
            if not isinstance(predicate_name, str) or not predicate_name:
                raise ValueError('Predicate name must be a non-empty string')
            if predicate_name in self._predicates:
                raise ValueError(
                    'Predicate {!r} is already registered'.format(predicate_name)
                )
            self._predicates[predicate_name] = fn
            return fn

        if func is not None:
            return do_register(func)
        return do_register

    def get(self, name: str) -> Callable[..., bool]:
        try:
            return self._predicates[name]
        except KeyError:
            raise UnknownOperatorError(
                'Unknown custom predicate: {!r}'.format(name)
            )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._predicates))


def _resolve_value(spec: Any, params: Mapping[str, Any]) -> Any:
    """
    Resolve a value specification to a snapshot.

    A ``{'$param': name}`` object pulls a value from the bound parameters;
    every other JSON-like value is used literally. All values are deep
    copied so later mutations of the caller's arguments cannot change the
    compiled query or its cache hash.
    """
    if (
        isinstance(spec, dict)
        and len(spec) == 1
        and _PARAM_KEY in spec
    ):
        name = spec[_PARAM_KEY]
        if not isinstance(name, str):
            raise InvalidQueryExpressionError(
                'Parameter reference must use a string name'
            )
        if name not in params:
            raise MissingParameterError(
                'Missing bound parameter: {!r}'.format(name)
            )
        return copy.deepcopy(params[name])

    return copy.deepcopy(spec)


def _validate_path(path: Any) -> tuple[str, ...]:
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(part, str) for part in path)
    ):
        raise InvalidQueryExpressionError(
            'Path must be a non-empty list of field name strings'
        )
    return tuple(path)


def _require_keys(
    node: Mapping,
    op: str,
    required: frozenset,
    optional: frozenset = frozenset(),
) -> None:
    keys = frozenset(node)
    missing = required - keys
    if missing:
        raise InvalidQueryExpressionError(
            'Operator {!r} is missing keys: {}'.format(
                op, ', '.join(sorted(missing))
            )
        )
    unknown = keys - required - optional - frozenset(('op',))
    if unknown:
        raise InvalidQueryExpressionError(
            'Operator {!r} got unknown keys: {}'.format(
                op, ', '.join(sorted(unknown))
            )
        )


def _compile(
    node: Any,
    params: Mapping[str, Any],
    registry: Optional[PredicateRegistry],
    remaining_depth: int,
    seen: set[int],
) -> QueryInstance:
    if remaining_depth < 0:
        raise InvalidQueryExpressionError(
            'Query expression exceeds the maximum nesting depth of {}'.format(
                MAX_EXPRESSION_DEPTH
            )
        )

    if not isinstance(node, Mapping):
        raise InvalidQueryExpressionError(
            'Each expression node must be a JSON object, got {}'.format(
                type(node).__name__
            )
        )

    node_id = id(node)
    if node_id in seen:
        raise InvalidQueryExpressionError('Query expression contains a cycle')
    seen.add(node_id)
    try:
        op = node.get('op')
        if not isinstance(op, str):
            raise InvalidQueryExpressionError(
                'Expression node is missing a string "op" key'
            )
        if op not in _BUILTIN_OPS:
            raise UnknownOperatorError('Unknown query operator: {!r}'.format(op))

        child_depth = remaining_depth - 1

        if op in _COMPARISON_OPS:
            _require_keys(node, op, frozenset(('path', 'value')))
            path = _validate_path(node['path'])
            rhs = _resolve_value(node['value'], params)
            field = _build_path(path)
            return _COMPARISON_BUILDERS[op](field, rhs)

        if op == 'exists':
            _require_keys(node, op, frozenset(('path',)))
            path = _validate_path(node['path'])
            return _build_path(path).exists()

        if op in ('matches', 'search'):
            _require_keys(
                node,
                op,
                frozenset(('path', 'pattern')),
                frozenset(('flags',)),
            )
            path = _validate_path(node['path'])
            pattern = node['pattern']
            flags = node.get('flags', 0)
            if not isinstance(pattern, str):
                raise InvalidQueryExpressionError(
                    'Operator {!r} requires a string pattern'.format(op)
                )
            if not isinstance(flags, int) or isinstance(flags, bool):
                raise InvalidQueryExpressionError(
                    'Operator {!r} requires integer regex flags'.format(op)
                )
            field = _build_path(path)
            if op == 'matches':
                return field.matches(pattern, flags)
            return field.search(pattern, flags)

        if op == 'one_of':
            _require_keys(node, op, frozenset(('path', 'values')))
            path = _validate_path(node['path'])
            values_spec = node['values']
            if isinstance(values_spec, Mapping) and len(values_spec) == 1 \
                    and _PARAM_KEY in values_spec:
                bound = _resolve_value(values_spec, params)
                if not isinstance(bound, list):
                    raise InvalidQueryExpressionError(
                        'Operator "one_of" parameter must bind to a list'
                    )
                values = bound
            elif isinstance(values_spec, list):
                values = [
                    _resolve_value(item, params)
                    for item in values_spec
                ]
            else:
                raise InvalidQueryExpressionError(
                    'Operator "one_of" requires a list of values'
                )
            return _build_path(path).one_of(values)

        if op in _UNARY_OPS:
            _require_keys(node, op, frozenset(('cond',)))
            cond = _compile(
                node['cond'], params, registry, child_depth, seen
            )
            return ~cond

        if op in _BINARY_OPS:
            _require_keys(node, op, frozenset(('left', 'right')))
            left = _compile(
                node['left'], params, registry, child_depth, seen
            )
            right = _compile(
                node['right'], params, registry, child_depth, seen
            )
            return left & right if op == 'and' else left | right

        if op in _ARRAY_OPS:
            _require_keys(node, op, frozenset(('path', 'cond')))
            path = _validate_path(node['path'])
            field = _build_path(path)
            cond = _compile_array_cond(
                node['cond'], params, registry, child_depth, seen
            )
            return field.any(cond) if op == 'any' else field.all(cond)

        # op == 'pred'
        _require_keys(
            node,
            op,
            frozenset(('path', 'name')),
            frozenset(('args',)),
        )
        if registry is None:
            raise UnknownOperatorError(
                'Custom predicates require a PredicateRegistry to be passed'
            )
        path = _validate_path(node['path'])
        name = node['name']
        if not isinstance(name, str):
            raise InvalidQueryExpressionError(
                'Operator "pred" requires a string predicate name'
            )
        args_spec = node.get('args', [])
        if not isinstance(args_spec, list):
            raise InvalidQueryExpressionError(
                'Operator "pred" requires "args" to be a list'
            )
        args = [_resolve_value(item, params) for item in args_spec]
        predicate = registry.get(name)
        return _build_path(path).test(predicate, *args)
    finally:
        seen.remove(node_id)


def _compile_array_cond(
    spec: Any,
    params: Mapping[str, Any],
    registry: Optional[PredicateRegistry],
    remaining_depth: int,
    seen: set[int],
) -> Union[QueryInstance, list[Any]]:
    if isinstance(spec, Mapping):
        if _PARAM_KEY in spec and len(spec) == 1:
            value = _resolve_value(spec, params)
            if isinstance(value, Mapping):
                return _compile(
                    value, params, registry, remaining_depth, seen
                )
            if isinstance(value, (list, tuple)):
                return list(value)
            raise InvalidQueryExpressionError(
                'An array condition parameter must bind to an expression or a '
                'list of values'
            )
        return _compile(spec, params, registry, remaining_depth, seen)

    if isinstance(spec, list):
        return [_resolve_value(item, params) for item in spec]

    raise InvalidQueryExpressionError(
        'Array condition must be an expression or a list of values'
    )


def _build_path(path: tuple[str, ...]) -> Query:
    query = Query()
    for part in path:
        query = query[part]
    return query


_COMPARISON_BUILDERS = {
    'eq': lambda field, value: field == value,
    'ne': lambda field, value: field != value,
    'lt': lambda field, value: field < value,
    'le': lambda field, value: field <= value,
    'gt': lambda field, value: field > value,
    'ge': lambda field, value: field >= value,
}


def compile_query_expression(
    expression: Mapping[str, Any],
    params: Optional[Mapping[str, Any]] = None,
    *,
    predicates: Optional[PredicateRegistry] = None,
    max_depth: int = MAX_EXPRESSION_DEPTH,
) -> QueryInstance:
    """
    Compile a serializable query expression into a native query instance.

    :param expression: The expression tree. Every node is a JSON object
                       containing an ``op`` key. Field paths are lists of
                       field name strings. Bound values are referenced with
                       ``{'$param': 'name'}``; plain JSON values may be used
                       inline as well.
    :param params: Values bound to ``$param`` references. Values are deep
                   copied when bound, so mutating them afterwards does not
                   change the compiled query or its cache hash.
    :param predicates: Registry of explicitly allowed custom predicates.
    :param max_depth: Maximum allowed nesting depth of the expression.
    :return: A :class:`~tinydb.queries.QueryInstance` equivalent to building
             the same query with :class:`~tinydb.queries.Query`.
    """
    if not isinstance(expression, Mapping):
        raise InvalidQueryExpressionError(
            'Query expression must be a JSON object, got {}'.format(
                type(expression).__name__
            )
        )
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise InvalidQueryExpressionError(
            'Bound parameters must be provided as a mapping'
        )
    if not all(isinstance(key, str) for key in params):
        raise InvalidQueryExpressionError(
            'Bound parameter names must be strings'
        )
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError('max_depth must be a non-negative integer')

    return _compile(expression, params, predicates, max_depth, set())
