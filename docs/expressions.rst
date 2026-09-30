Storing Queries as JSON
=======================

.. versionadded:: 4.10

Sometimes queries need to be stored — for example in a configuration file
or in a database — and restored later. Since pickling user-supplied
functions is unsafe, TinyDB provides a limited, JSON-serializable query
expression format in :mod:`tinydb.expressions`.

An expression is a plain JSON object with an ``op`` field:

>>> from tinydb.expressions import QueryExpression
>>> expr = QueryExpression.parse({
...     'op': 'and',
...     'args': [
...         {'op': 'ge', 'path': ['user', 'age'], 'value': {'$param': 'min'}},
...         {'op': 'any', 'path': 'tags', 'values': ['admin', 'staff']},
...     ],
... })

Expressions are validated when parsed: malformed structure, excessive
nesting depth, unknown operators or predicates and (when ``params`` are
passed to :meth:`~tinydb.expressions.QueryExpression.parse`) missing
parameters all raise :class:`~tinydb.expressions.ExpressionError`.

Binding Parameters
------------------

Values are bound independently from the expression itself. Anywhere a
value is expected, ``{'$param': 'name'}`` refers to a parameter that is
supplied when the expression is bound to a query:

>>> query = expr.bind({'min': 18})
>>> db.search(query)  # doctest: +SKIP

Parameter values are deep-copied at bind time, so mutating a mutable
parameter afterwards changes neither the query's behavior nor its cache
hash. The required parameter names are available as
:attr:`~tinydb.expressions.QueryExpression.required_params`.

Storing and Restoring
---------------------

Use :meth:`~tinydb.expressions.QueryExpression.to_json` (or
:meth:`~tinydb.expressions.QueryExpression.to_dict`) to store an
expression and :meth:`~tinydb.expressions.QueryExpression.from_json` to
restore it:

>>> stored = expr.to_json()
>>> restored = QueryExpression.from_json(stored)
>>> from tinydb import Query
>>> restored.bind({'min': 18}) == ((Query()['user']['age'] >= 18)
...                                & Query()['tags'].any(['admin', 'staff']))
True

A restored query behaves exactly like a directly constructed
:class:`~tinydb.queries.Query` — including missing fields, nested arrays
and the query cache hash. For one-shot loading without keeping the
expression around, use :func:`tinydb.expressions.load_query`.

Operators
---------

.. list-table::
   :header-rows: 1

   * - Operator
     - Fields
     - Equivalent
   * - ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``
     - ``path``, ``value``
     - ``where(path) == value`` etc.
   * - ``exists``
     - ``path``
     - ``where(path).exists()``
   * - ``matches``, ``search``
     - ``path``, ``pattern``, optional ``flags``
     - ``where(path).matches(pattern)`` etc.
   * - ``one_of``
     - ``path``, ``values``
     - ``where(path).one_of(values)``
   * - ``any``, ``all``
     - ``path`` plus ``where`` (a nested expression evaluated for each
       element) or ``values`` (a list)
     - ``where(path).any(...)`` etc.
   * - ``and``, ``or``
     - ``args`` (a non-empty list of expressions)
     - ``q1 & q2``, ``q1 | q2``
   * - ``not``
     - ``arg``
     - ``~q``
   * - ``noop``
     - —
     - ``Query().noop()``
   * - ``pred``
     - ``path``, ``name``, optional ``args``
     - ``where(path).test(func, *args)``

``path`` is a field name or a list of nested field names. An empty path
(``[]``) addresses the document or array element itself, which allows
conditions on arrays of scalars.

Custom Predicates and Operators
-------------------------------

Predicates and operators are resolved exclusively from an explicit
:class:`~tinydb.expressions.OperatorRegistry` — expressions can only
reference what has been registered by name, and parsing never imports or
executes arbitrary code:

>>> from tinydb.expressions import OperatorRegistry
>>> registry = OperatorRegistry.with_defaults()
>>> registry.register_predicate('is_even', lambda value: value % 2 == 0)
>>> expr = QueryExpression.parse(
...     {'op': 'pred', 'path': 'n', 'name': 'is_even'},
...     registry=registry)
>>> expr.bind()({'n': 4})
True

Custom operators can be added with
:meth:`~tinydb.expressions.OperatorRegistry.register_operator`. The
handler receives the expression node and a compiler context (with
``compile``, ``resolve`` and ``query`` helpers) and returns a query
instance:

>>> def between(node, ctx):
...     query = ctx.query(node['path'])
...     return (query >= ctx.resolve(node['low'])) & \
...            (query <= ctx.resolve(node['high']))
>>> registry.register_operator('between', between)
>>> expr = QueryExpression.parse(
...     {'op': 'between', 'path': 'age', 'low': 18, 'high': 30},
...     registry=registry)
>>> expr.bind() == ((Query()['age'] >= 18) & (Query()['age'] <= 30))
True
