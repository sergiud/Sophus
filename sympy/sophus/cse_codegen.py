# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

import io
import math
import os
import random
import re

import sympy

# Setting this environment variable to 1 makes the tests rewrite the
# generated C++ code in cpp_gencode/ instead of checking it, for example with
#
#   SOPHUS_REGENERATE_CPP_GENCODE=1 ./run_tests.sh
REGENERATE_VARIABLE = 'SOPHUS_REGENERATE_CPP_GENCODE'

# The number of sample points at which the generated code is evaluated.
NUM_SAMPLES = 3

# The relative tolerance of the generated code with respect to the symbolic
# expression. Both are evaluated in double precision, but the common
# subexpressions round differently than the symbolic expression.
RELATIVE_TOLERANCE = 1e-9

# The functions that the generated code calls.
FUNCTIONS = {
    'atan': math.atan,
    'atan2': math.atan2,
    'cos': math.cos,
    'pow': math.pow,
    'sin': math.sin,
    'sqrt': math.sqrt,
}


def cse_codegen(symbols):
    cse_results = sympy.cse(symbols, sympy.numbered_symbols("c"))
    output = io.StringIO()
    for helper in cse_results[0]:
        output.write("Scalar const ")
        output.write(sympy.printing.ccode(helper[1], helper[0]))
        output.write("\n")
    assert len(cse_results[1]) == 1

    output.write(sympy.printing.ccode(cse_results[1][0], "result"))
    output.write("\n")
    output.seek(0)
    return output


def evaluate_cpp_gencode(code, values):
    """Evaluates generated C++ code at the given values of its symbols.

    The code consists of assignments such as ``Scalar const c0 = sin(x);``
    and ``result[0] = c0;``, as ``cse_codegen`` writes them. Symbols appear
    in the code under their names, such as ``q.w()`` or ``omega[0]``.

    Returns the values assigned to ``result`` as a list, or a list with a
    single element if ``result`` is a scalar.
    """
    placeholders = {}
    # Longer names first, so that a name is not replaced inside a longer one.
    names = sorted(values, key=lambda symbol: -len(symbol.name))
    for index, symbol in enumerate(names):
        placeholder = '_symbol{}'.format(index)
        placeholders[placeholder] = values[symbol]
        # A name must not continue another name, so that the symbol t[1] does
        # not match within result[1].
        pattern = r'(?<![\w.]){}'.format(re.escape(symbol.name))
        if re.search(r'\w$', symbol.name):
            pattern += r'\b'
        code = re.sub(pattern, placeholder, code)

    namespace = dict(FUNCTIONS)
    namespace.update(placeholders)
    namespace['result'] = {}
    for line in code.splitlines():
        statement = line.strip()
        if not statement:
            continue
        statement = re.sub(r'^Scalar const ', '', statement)
        statement = re.sub(r';$', '', statement)
        if statement.startswith('result ='):
            statement = statement.replace('result =', 'result[0] =', 1)
        exec(statement, namespace)

    result = namespace['result']
    return [result[index] for index in sorted(result)]


def evaluate_expression(expression, values):
    """Evaluates a symbolic scalar or matrix in row-major order."""
    entries = list(expression) if isinstance(
        expression, sympy.MatrixBase) else [expression]
    return [float(sympy.sympify(entry).subs(values).evalf())
            for entry in entries]


def check_cpp_gencode(test_case, expression, filename):
    """Checks that the generated C++ code in ``filename`` computes
    ``expression``.

    The generated code is compared by its values at sample points rather than
    by its text, since sympy chooses different but equivalent common
    subexpressions in different versions. If the environment variable named
    by ``REGENERATE_VARIABLE`` is set to 1, the code is regenerated instead.
    """
    if os.environ.get(REGENERATE_VARIABLE) == '1':
        with open(filename, 'w') as file:
            file.write(cse_codegen(expression).getvalue())
        return

    with open(filename) as file:
        code = file.read()

    symbols = sorted(expression.free_symbols, key=lambda symbol: symbol.name)
    rng = random.Random(filename)
    for sample in range(NUM_SAMPLES):
        values = {symbol: rng.uniform(0.2, 0.8) for symbol in symbols}
        expected = evaluate_expression(expression, values)
        actual = evaluate_cpp_gencode(code, values)
        test_case.assertEqual(
            len(actual), len(expected),
            '{} assigns {} results, but the expression has {} entries'.format(
                filename, len(actual), len(expected)))
        for index, (a, e) in enumerate(zip(actual, expected)):
            test_case.assertTrue(
                math.isclose(a, e, rel_tol=RELATIVE_TOLERANCE,
                             abs_tol=RELATIVE_TOLERANCE),
                '{} result[{}] is {}, but the expression is {} at {}'.format(
                    filename, index, a, e, values))
