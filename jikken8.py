from sympy import Symbol
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')
from sympy import symbols

z = Symbol('zeta')
s = Symbol('sigma')
a,e,b = symbols('alpha eta beta')
