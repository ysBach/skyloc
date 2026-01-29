"""Utility functions for filtering and operator handling."""

import pyarrow.dataset as ds

SUPPORTED_OPERATORS = (">", ">=", "<", "<=", "==", "!=", "in", "not in")


def validate_operator(op, col=None):
    """Validate that operator is supported.

    Parameters
    ----------
    op : str
        Operator to validate
    col : str, optional
        Column name for error message context

    Raises
    ------
    ValueError
        If operator is not supported
    """
    if op not in SUPPORTED_OPERATORS:
        if col:
            raise ValueError(
                f"Unsupported operator '{op}' for column '{col}'. "
                f"Supported: {', '.join(repr(o) for o in SUPPORTED_OPERATORS)}"
            )
        else:
            raise ValueError(
                f"Unsupported operator: {op}. "
                f"Supported: {', '.join(repr(o) for o in SUPPORTED_OPERATORS)}"
            )


def apply_filter_operator(op, left, right):
    """Apply a filter operator to left and right operands.

    This function abstracts the repeated if/elif chains for operator
    application, supporting both PyArrow fields and pandas Series/arrays.

    Parameters
    ----------
    op : str
        Operator: '>', '>=', '<', '<=', '==', '!=', 'in', 'not in'
    left : pyarrow.Field, pandas.Series, or array-like
        Left operand (field or column)
    right : scalar or array-like
        Right operand (value or values)

    Returns
    -------
    result : pyarrow.Expression, pandas.Series, or boolean array
        Result of applying the operator

    Raises
    ------
    ValueError
        If operator is not supported

    Examples
    --------
    PyArrow usage::

        field = ds.field('vmag')
        expr = apply_filter_operator('>', field, 20)

    Pandas usage::

        mask = apply_filter_operator('<', df['vmag'], 20)
        filtered = df[mask]
    """
    if op == ">":
        return left > right
    elif op == ">=":
        return left >= right
    elif op == "<":
        return left < right
    elif op == "<=":
        return left <= right
    elif op == "==":
        return left == right
    elif op == "!=":
        return left != right
    elif op == "in":
        if hasattr(left, "isin"):
            return left.isin(right)
        else:
            raise TypeError(f"'in' operator requires isin() method, got {type(left)}")
    elif op == "not in":
        if hasattr(left, "isin"):
            return ~left.isin(right)
        else:
            raise TypeError(
                f"'not in' operator requires isin() method, got {type(left)}"
            )
    else:
        validate_operator(op)


def to_numeric_if_possible(value_str):
    """Convert string to numeric type if possible, preferring int over float.

    Parameters
    ----------
    value_str : str
        String value to convert

    Returns
    -------
    numeric or str
        Integer if value is integer-like float, float if decimal, original string otherwise

    Examples
    --------
    >>> to_numeric_if_possible("42")
    42
    >>> to_numeric_if_possible("3.14")
    3.14
    >>> to_numeric_if_possible("foo")
    'foo'
    """
    try:
        numeric_val = float(value_str)
        if numeric_val.is_integer():
            return int(numeric_val)
        return numeric_val
    except ValueError:
        return value_str
