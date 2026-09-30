"""
JSON-serializable query expressions.

This module implements a limited, JSON-serializable format for TinyDB
queries so that query conditions can be stored (e.g. in a config file or
a database) and restored later without using ``pickle`` on user-supplied
functions.

An expression is a plain JSON object with an ``op`` field, for example::

    {
        "op": "and",
        "args": [
            {"op": "eq", "path": ["user", "name"],
             "value": {"$param": "name"}},
            {"op": "any", "path": "tags", "values": ["admin", "staff"]}
        ]
    }

Values are bound independently from the expression itself: anywhere a
value is expected, ``{"$param": "name"}`` refers to a parameter that is
supplied when the expression is bound to a query. Operators and custom
predicates are resolved exclusively from an explicit
:class:`OperatorRegistry`; parsing never imports or executes arbitrary
code.

Parsing validates the expression's structure, its nesting depth, unknown
operators/predicates and (when parameters are supplied) missing
parameters. Binding deep-copies all parameter values, so mutating a
mutable parameter after binding changes neither the query's behavior nor
its cache hash.

Usage:

>>> from tinydb.queries import Query
>>> from tinydb.expressions import QueryExpression
>>> expr = QueryExpression.parse({'op': 'eq', 'path': 'a', 'value': 1})
>>> query = expr.bind()
>>> query == (Query()['a'] == 1)
True
"""

import copy
import json
import re
from typing import Any, Callable, Optional, Union

from .queries import Query, QueryInstance, is_sequence
from .utils import freeze

__all__ = (
    'ExpressionError',
    'OperatorRegistry',
    'QueryExpression',
    'default_registry',
    'load_query',
)

#: Default maximum nesting depth accepted when parsing an expression.
DEFAULT_MAX_DEPTH = 32

#: Key that marks a parameter reference inside an expression.
PARAM_REF_KEY = '$param'

#: Handler for a registered operator. Receives the expression node and
#: the bind-time compiler context and returns a query instance.
OperatorHandler = Callable[[dict, '_Compiler'], 'QueryInstance']


class ExpressionError(ValueError):
    """
    Raised when a query expression is invalid.

    This covers malformed structure, excessive nesting depth, unknown
    operators or predicates and missing parameters.
    """


def _is_param_ref(value: Any) -> bool:
    return isinstance(value, dict) and PARAM_REF_KEY in value


def _resolve_value(value: Any, params: dict) -> Any:
    """
    Substitute parameter references in a literal value.

    Parameter values are deep-copied so that mutating the original
    parameter after binding cannot change the bound query's behavior or
    its cache hash.
    """
    if isinstance(value, dict):
        if PARAM_REF_KEY in value:
            name = value[PARAM_REF_KEY]
            if name not in params:
                raise ExpressionError(f'missing parameter: {name!r}')

            return copy.deepcopy(params[name])

        return {key: _resolve_value(item, params)
                for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_value(item, params) for item in value]

    return value


def _generate(query: Query, test: Callable, hashval: tuple) -> QueryInstance:
    """
    Generate a query exactly like the corresponding ``Query`` method does,
    but additionally allow an empty path (which addresses the document or
    array element itself).
    """
    return query._generate_test(test, hashval, allow_empty_path=True)


class OperatorRegistry:
    """
    An explicit registry of the operators and predicates that query
    expressions are allowed to use.

    Expressions can only use operators and predicates that have been
    registered here, referenced by name. Nothing is ever imported or
    looked up by module path, so a stored expression cannot trigger
    arbitrary code execution.

    Use :meth:`with_defaults` to create a registry pre-populated with the
    built-in operators, then register custom predicates (and operators)
    on top of it:

    >>> registry = OperatorRegistry.with_defaults()
    >>> registry.register_predicate('is_even', lambda value: value % 2 == 0)
    """

    def __init__(self) -> None:
        self._operators: dict = {}
        self._predicates: dict = {}

    @classmethod
    def with_defaults(cls) -> 'OperatorRegistry':
        """
        Create a registry containing all built-in operators.
        """
        registry = cls()
        for name, handler in _BUILTIN_OPERATORS.items():
            registry.register_operator(name, handler)

        return registry

    def register_operator(self, name: str, handler: OperatorHandler) -> None:
        """
        Register a custom operator.

        The handler is called as ``handler(node, compiler)`` when an
        expression with ``{"op": name, ...}`` is bound and must return a
        :class:`~tinydb.queries.QueryInstance`. Use ``compiler.compile``
        for nested expressions, ``compiler.resolve`` for values that may
        contain parameter references and ``compiler.query`` to build a
        query for a field path. Handlers should raise
        :class:`ExpressionError` on invalid node structure.

        :param name: The operator name used in expressions.
        :param handler: The handler building the query.
        """
        if not isinstance(name, str) or not name:
            raise ValueError('operator name must be a non-empty string')

        if not callable(handler):
            raise ValueError('operator handler must be callable')

        self._operators[name] = handler

    def register_predicate(self, name: str, func: Callable) -> None:
        """
        Register a custom predicate for use with the ``pred`` operator.

        The predicate is called with the resolved field value as the
        first argument, followed by any ``args`` from the expression,
        mirroring :meth:`tinydb.queries.Query.test`.

        :param name: The predicate name used in expressions.
        :param func: The predicate function.
        """
        if not isinstance(name, str) or not name:
            raise ValueError('predicate name must be a non-empty string')

        if not callable(func):
            raise ValueError('predicate must be callable')

        self._predicates[name] = func

    def get_operator(self, name: str) -> Optional[OperatorHandler]:
        return self._operators.get(name)

    def get_predicate(self, name: str) -> Optional[Callable]:
        return self._predicates.get(name)

    @property
    def operators(self) -> frozenset:
        """The names of all registered operators."""
        return frozenset(self._operators)

    @property
    def predicates(self) -> frozenset:
        """The names of all registered predicates."""
        return frozenset(self._predicates)


class _Compiler:
    """
    Bind-time compiler turning a validated expression tree into a
    :class:`~tinydb.queries.QueryInstance`.

    This is also the context object passed to custom operator handlers.
    """

    def __init__(self, registry: OperatorRegistry, params: dict,
                 max_depth: int) -> None:
        self.registry = registry
        self._params = params
        self._max_depth = max_depth
        self._depth = 0

    def compile(self, node: dict) -> QueryInstance:
        """
        Compile a (sub-)expression node into a query instance.
        """
        self._depth += 1
        try:
            if self._depth > self._max_depth:
                raise ExpressionError(
                    f'expression exceeds maximum depth of {self._max_depth}'
                )

            op = node['op']
            handler = self.registry.get_operator(op)
            if handler is None:
                raise ExpressionError(f'unknown operator: {op!r}')

            return handler(node, self)
        finally:
            self._depth -= 1

    def resolve(self, value: Any) -> Any:
        """
        Resolve parameter references in a literal value.
        """
        return _resolve_value(value, self._params)

    def query(self, path: Union[str, list] = ()) -> Query:
        """
        Build a :class:`~tinydb.queries.Query` for a field path.
        """
        if isinstance(path, str):
            path = [path]

        query = Query()
        for segment in path:
            query = query[segment]

        return query


# --- Built-in operator handlers ---------------------------------------------
#
# Each handler mirrors the corresponding ``Query`` method exactly (same test
# function and same hash value), so a restored query is indistinguishable
# from a directly constructed one, including its cache behavior.

def _compile_eq(node: dict, ctx: _Compiler) -> QueryInstance:
    rhs = ctx.resolve(node['value'])
    query = ctx.query(node['path'])
    return _generate(query, lambda value: value == rhs,
                     ('==', query._path, freeze(rhs)))


def _compile_ne(node: dict, ctx: _Compiler) -> QueryInstance:
    rhs = ctx.resolve(node['value'])
    query = ctx.query(node['path'])
    return _generate(query, lambda value: value != rhs,
                     ('!=', query._path, freeze(rhs)))


def _make_ordering(op_symbol: str, compare: Callable) -> OperatorHandler:
    def handler(node: dict, ctx: _Compiler) -> QueryInstance:
        rhs = ctx.resolve(node['value'])
        query = ctx.query(node['path'])
        return _generate(query, lambda value: compare(value, rhs),
                         (op_symbol, query._path, rhs))

    return handler


_compile_lt = _make_ordering('<', lambda value, rhs: value < rhs)
_compile_le = _make_ordering('<=', lambda value, rhs: value <= rhs)
_compile_gt = _make_ordering('>', lambda value, rhs: value > rhs)
_compile_ge = _make_ordering('>=', lambda value, rhs: value >= rhs)


def _compile_exists(node: dict, ctx: _Compiler) -> QueryInstance:
    query = ctx.query(node['path'])
    return _generate(query, lambda _: True, ('exists', query._path))


def _make_regex(op_name: str, match_func: Callable) -> OperatorHandler:
    def handler(node: dict, ctx: _Compiler) -> QueryInstance:
        regex = ctx.resolve(node['pattern'])
        flags = ctx.resolve(node.get('flags', 0))

        if not isinstance(regex, str):
            raise ExpressionError(
                f"operator {op_name!r}: 'pattern' must resolve to a string"
            )

        if not isinstance(flags, int):
            raise ExpressionError(
                f"operator {op_name!r}: 'flags' must resolve to an integer"
            )

        def test(value):
            if not isinstance(value, str):
                return False

            return match_func(regex, value, flags) is not None

        query = ctx.query(node['path'])
        return _generate(query, test, (op_name, query._path, regex))

    return handler


_compile_matches = _make_regex('matches', re.match)
_compile_search = _make_regex('search', re.search)


def _compile_one_of(node: dict, ctx: _Compiler) -> QueryInstance:
    items = ctx.resolve(node['values'])

    if not isinstance(items, list):
        raise ExpressionError(
            "operator 'one_of': 'values' must resolve to a list"
        )

    query = ctx.query(node['path'])
    return _generate(query, lambda value: value in items,
                     ('one_of', query._path, freeze(items)))


def _make_quantifier(op_name: str, aggregate: Callable) -> OperatorHandler:
    def handler(node: dict, ctx: _Compiler) -> QueryInstance:
        if 'where' in node:
            cond: Any = ctx.compile(node['where'])
        else:
            cond = ctx.resolve(node['values'])
            if not isinstance(cond, list):
                raise ExpressionError(
                    f"operator {op_name!r}: 'values' must resolve to a list"
                )

        if callable(cond):
            def test(value):
                return is_sequence(value) and aggregate(
                    cond(element) for element in value
                )
        elif op_name == 'any':
            def test(value):
                return is_sequence(value) and any(
                    element in cond for element in value
                )
        else:
            def test(value):
                return is_sequence(value) and all(
                    element in value for element in cond
                )

        query = ctx.query(node['path'])
        return _generate(query, test, (op_name, query._path, freeze(cond)))

    return handler


_compile_any = _make_quantifier('any', any)
_compile_all = _make_quantifier('all', all)


def _make_logical(combine: Callable) -> OperatorHandler:
    def handler(node: dict, ctx: _Compiler) -> QueryInstance:
        args = [ctx.compile(arg) for arg in node['args']]
        result = args[0]
        for arg in args[1:]:
            result = combine(result, arg)

        return result

    return handler


_compile_and = _make_logical(lambda left, right: left & right)
_compile_or = _make_logical(lambda left, right: left | right)


def _compile_not(node: dict, ctx: _Compiler) -> QueryInstance:
    return ~ctx.compile(node['arg'])


def _compile_noop(node: dict, ctx: _Compiler) -> QueryInstance:
    return Query().noop()


def _compile_pred(node: dict, ctx: _Compiler) -> QueryInstance:
    name = node['name']
    func = ctx.registry.get_predicate(name)
    if func is None:
        raise ExpressionError(f'unknown predicate: {name!r}')

    args = [ctx.resolve(arg) for arg in node.get('args', [])]
    query = ctx.query(node['path'])
    return _generate(
        query,
        lambda value: func(value, *args),
        ('test', query._path, func, tuple(freeze(arg) for arg in args))
    )


_BUILTIN_OPERATORS: dict = {
    'eq': _compile_eq,
    'ne': _compile_ne,
    'lt': _compile_lt,
    'le': _compile_le,
    'gt': _compile_gt,
    'ge': _compile_ge,
    'exists': _compile_exists,
    'matches': _compile_matches,
    'search': _compile_search,
    'one_of': _compile_one_of,
    'any': _compile_any,
    'all': _compile_all,
    'and': _compile_and,
    'or': _compile_or,
    'not': _compile_not,
    'noop': _compile_noop,
    'pred': _compile_pred,
}


# --- Parse-time validation ---------------------------------------------------

class _Validator:
    """
    Parse-time validator for expression trees.

    Checks the structure of built-in operators, collects referenced
    parameter names and enforces the nesting depth limit. For custom
    operators only generic value checks (JSON-safety, parameter
    references, depth) are possible here; their node structure is
    validated by their handler when the expression is bound.
    """

    def __init__(self, registry: OperatorRegistry, max_depth: int) -> None:
        self.registry = registry
        self.max_depth = max_depth
        self.params: set = set()

    def validate_expression(self, node: Any, depth: int = 1) -> dict:
        if depth > self.max_depth:
            raise ExpressionError(
                f'expression exceeds maximum depth of {self.max_depth}'
            )

        if not isinstance(node, dict):
            raise ExpressionError(
                'expression must be an object with an "op" field, '
                f'got {type(node).__name__}'
            )

        op = node.get('op')
        if not isinstance(op, str):
            raise ExpressionError(
                'expression must have an "op" field of type string'
            )

        if self.registry.get_operator(op) is None:
            raise ExpressionError(f'unknown operator: {op!r}')

        validator = _BUILTIN_VALIDATORS.get(op)
        if validator is not None:
            return validator(self, node, depth)

        # Custom operator: validate everything we can generically and keep
        # the node as-is for the handler to interpret.
        self.scan_value(node, depth)
        return dict(node)

    def scan_value(self, value: Any, depth: int) -> None:
        if depth > self.max_depth:
            raise ExpressionError(
                f'expression exceeds maximum depth of {self.max_depth}'
            )

        if isinstance(value, dict):
            if PARAM_REF_KEY in value:
                if set(value.keys()) != {PARAM_REF_KEY}:
                    raise ExpressionError(
                        f'parameter reference must be an object with only '
                        f'the {PARAM_REF_KEY!r} key'
                    )

                name = value[PARAM_REF_KEY]
                if not isinstance(name, str):
                    raise ExpressionError(
                        'parameter name must be a string'
                    )

                self.params.add(name)
                return

            for key, item in value.items():
                if not isinstance(key, str):
                    raise ExpressionError('object keys must be strings')

                self.scan_value(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.scan_value(item, depth + 1)
        elif value is None or isinstance(value, (str, int, float, bool)):
            return
        else:
            raise ExpressionError(
                f'unsupported value of type {type(value).__name__}: '
                'expressions must be JSON-serializable'
            )

    def parse_path(self, node: dict, op: str) -> list:
        path = node['path']

        if isinstance(path, str):
            return [path]

        if isinstance(path, list) and all(
                isinstance(segment, str) for segment in path):
            return list(path)

        raise ExpressionError(
            f"operator {op!r}: 'path' must be a string or a list of strings"
        )


def _check_keys(op: str, node: dict, required: tuple,
                optional: tuple = ()) -> None:
    keys = set(node.keys()) - {'op'}

    missing = [key for key in required if key not in keys]
    if missing:
        raise ExpressionError(
            f"operator {op!r}: missing field(s): {', '.join(missing)}"
        )

    unexpected = keys - set(required) - set(optional)
    if unexpected:
        raise ExpressionError(
            f"operator {op!r}: unexpected field(s): "
            f"{', '.join(sorted(unexpected))}"
        )


def _validate_comparison(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path', 'value'))
    ctx.scan_value(node['value'], depth)
    return {'op': op, 'path': ctx.parse_path(node, op),
            'value': node['value']}


def _validate_exists(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path',))
    return {'op': op, 'path': ctx.parse_path(node, op)}


def _validate_regex(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path', 'pattern'), optional=('flags',))

    pattern = node['pattern']
    ctx.scan_value(pattern, depth)
    if not isinstance(pattern, str) and not _is_param_ref(pattern):
        raise ExpressionError(
            f"operator {op!r}: 'pattern' must be a string or "
            'a parameter reference'
        )

    flags = node.get('flags', 0)
    ctx.scan_value(flags, depth)
    if not isinstance(flags, int) and not _is_param_ref(flags):
        raise ExpressionError(
            f"operator {op!r}: 'flags' must be an integer or "
            'a parameter reference'
        )

    return {'op': op, 'path': ctx.parse_path(node, op),
            'pattern': pattern, 'flags': flags}


def _validate_values_field(ctx: _Validator, node: dict, depth: int,
                           op: str) -> None:
    values = node['values']
    ctx.scan_value(values, depth)
    if not isinstance(values, list) and not _is_param_ref(values):
        raise ExpressionError(
            f"operator {op!r}: 'values' must be a list or "
            'a parameter reference'
        )


def _validate_one_of(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path', 'values'))
    _validate_values_field(ctx, node, depth, op)
    return {'op': op, 'path': ctx.parse_path(node, op),
            'values': node['values']}


def _validate_quantifier(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path',), optional=('where', 'values'))

    has_where = 'where' in node
    has_values = 'values' in node
    if has_where == has_values:
        raise ExpressionError(
            f"operator {op!r}: exactly one of 'where' and 'values' "
            'is required'
        )

    result = {'op': op, 'path': ctx.parse_path(node, op)}
    if has_where:
        result['where'] = ctx.validate_expression(node['where'], depth + 1)
    else:
        _validate_values_field(ctx, node, depth, op)
        result['values'] = node['values']

    return result


def _validate_logical(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('args',))

    args = node['args']
    if not isinstance(args, list) or not args:
        raise ExpressionError(
            f"operator {op!r}: 'args' must be a non-empty list"
        )

    return {'op': op,
            'args': [ctx.validate_expression(arg, depth + 1)
                     for arg in args]}


def _validate_not(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('arg',))
    return {'op': op, 'arg': ctx.validate_expression(node['arg'], depth + 1)}


def _validate_noop(ctx: _Validator, node: dict, depth: int) -> dict:
    _check_keys(node['op'], node, required=())
    return {'op': 'noop'}


def _validate_pred(ctx: _Validator, node: dict, depth: int) -> dict:
    op = node['op']
    _check_keys(op, node, required=('path', 'name'), optional=('args',))

    name = node['name']
    if not isinstance(name, str):
        raise ExpressionError(
            f"operator {op!r}: 'name' must be a string"
        )

    if ctx.registry.get_predicate(name) is None:
        raise ExpressionError(f'unknown predicate: {name!r}')

    args = node.get('args', [])
    if not isinstance(args, list):
        raise ExpressionError(
            f"operator {op!r}: 'args' must be a list"
        )

    ctx.scan_value(args, depth)
    return {'op': op, 'path': ctx.parse_path(node, op),
            'name': name, 'args': args}


_BUILTIN_VALIDATORS: dict = {
    'eq': _validate_comparison,
    'ne': _validate_comparison,
    'lt': _validate_comparison,
    'le': _validate_comparison,
    'gt': _validate_comparison,
    'ge': _validate_comparison,
    'exists': _validate_exists,
    'matches': _validate_regex,
    'search': _validate_regex,
    'one_of': _validate_one_of,
    'any': _validate_quantifier,
    'all': _validate_quantifier,
    'and': _validate_logical,
    'or': _validate_logical,
    'not': _validate_not,
    'noop': _validate_noop,
    'pred': _validate_pred,
}


# --- Public API --------------------------------------------------------------

class QueryExpression:
    """
    A validated, JSON-serializable query expression.

    Use :meth:`parse` (or :meth:`from_json`) to validate an expression and
    :meth:`bind` to compile it into a
    :class:`~tinydb.queries.QueryInstance` by binding its parameters.
    The resulting query behaves exactly like a directly constructed
    :class:`~tinydb.queries.Query`, including its cache hash.

    Use :meth:`to_dict`/:meth:`to_json` to store the expression and
    :meth:`parse`/:meth:`from_json` to restore it later.
    """

    def __init__(self, tree: dict, registry: OperatorRegistry,
                 max_depth: int, required_params: frozenset) -> None:
        self._tree = tree
        self._registry = registry
        self._max_depth = max_depth
        self._required_params = required_params

    @classmethod
    def parse(cls, data: dict, *,
              registry: Optional[OperatorRegistry] = None,
              max_depth: int = DEFAULT_MAX_DEPTH,
              params: Optional[dict] = None) -> 'QueryExpression':
        """
        Validate an expression given as a JSON-compatible dict.

        :param data: The expression (a dict with an ``op`` field).
        :param registry: The registry to resolve operators and predicates
                         from. Defaults to :func:`default_registry`.
        :param max_depth: The maximum allowed nesting depth.
        :param params: If given, check that all referenced parameters are
                       present in this mapping.
        :raises ExpressionError: If the expression is invalid.
        """
        if registry is None:
            registry = default_registry()

        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError('max_depth must be a positive integer')

        validator = _Validator(registry, max_depth)
        tree = copy.deepcopy(validator.validate_expression(data))
        expression = cls(tree, registry, max_depth,
                         frozenset(validator.params))

        if params is not None:
            expression._check_params(params)

        return expression

    @classmethod
    def from_json(cls, data: Union[str, bytes], **kwargs) -> 'QueryExpression':
        """
        Parse an expression from its JSON representation.
        """
        return cls.parse(json.loads(data), **kwargs)

    @property
    def required_params(self) -> frozenset:
        """The names of all parameters referenced by this expression."""
        return self._required_params

    def bind(self, params: Optional[dict] = None,
             **named_params: Any) -> QueryInstance:
        """
        Bind the expression's parameters and compile it to a query.

        Parameter values are deep-copied, so mutating a mutable parameter
        after binding changes neither the query's behavior nor its cache
        hash.

        :param params: A mapping of parameter names to values.
        :param named_params: Parameters may also be passed as keyword
                             arguments.
        :raises ExpressionError: If a referenced parameter is missing.
        :return: A :class:`~tinydb.queries.QueryInstance`.
        """
        bound = dict(params) if params is not None else {}
        bound.update(named_params)
        self._check_params(bound)

        compiler = _Compiler(self._registry, bound, self._max_depth)
        return compiler.compile(self._tree)

    def to_dict(self) -> dict:
        """
        Return the expression as a JSON-serializable dict.
        """
        return copy.deepcopy(self._tree)

    def to_json(self, **kwargs: Any) -> str:
        """
        Serialize the expression to a JSON string.
        """
        return json.dumps(self._tree, **kwargs)

    def _check_params(self, params: dict) -> None:
        missing = self._required_params - params.keys()
        if missing:
            raise ExpressionError(
                f"missing parameter(s): {', '.join(sorted(missing))}"
            )

    def __repr__(self) -> str:
        return f'QueryExpression({self._tree!r})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QueryExpression):
            return self._tree == other._tree

        return False


_default_registry: Optional[OperatorRegistry] = None


def default_registry() -> OperatorRegistry:
    """
    Return the shared default registry containing all built-in operators.

    Custom predicates and operators registered here are available to all
    expressions parsed without an explicit registry.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = OperatorRegistry.with_defaults()

    return _default_registry


def load_query(data: dict, params: Optional[dict] = None, *,
               registry: Optional[OperatorRegistry] = None,
               max_depth: int = DEFAULT_MAX_DEPTH,
               **named_params: Any) -> QueryInstance:
    """
    Parse an expression and bind it to a query in one step.

    This is a shorthand for ``QueryExpression.parse(...).bind(...)``.
    """
    expression = QueryExpression.parse(data, registry=registry,
                                       max_depth=max_depth)
    return expression.bind(params, **named_params)
