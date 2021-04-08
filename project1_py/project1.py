#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

def local_descent(x, alpha, d):
    x_next = x + alpha*d
    return x_next

def gradient_descent(x, alpha, g):
    g = g(x)
    d = -g/np.linalg.norm(g)
    x_next = local_descent(x, alpha, d)
    return x_next

def backtrack_line_search(f, g, x, d, alpha, p = .005, B = 1e-12):
    y = f(x)
    while(f(x + alpha*d) > y + B*alpha*(np.dot(g,d))):
        alpha *= p
    return alpha

def gradient_descent_with_line_search(f, x, alpha, g):
    g = g(x)
    d = -g/np.linalg.norm(g)
    alpha = backtrack_line_search(f, g, x, d, alpha)
    x_next = local_descent(x, alpha, d)
    return x_next

def momentum(g, x, v, alpha, B): 
    g = g(x)
    norm_g = g/np.linalg.norm(g)
    v_next = B*v - alpha*norm_g
    x_next = x + v_next
    return x_next, v_next

def nesterov_momentum(g, x, v, alpha, B):
    gr = g(x + B*v)
    norm_gr = gr/np.linalg.norm(gr)
    v_next = B*v - alpha*norm_gr
    x_next = x + v_next
    return x_next, v_next



def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """

    x_last = x0
    v_last = np.zeros(len(x0))
    # x_history = []
    while(count() < n):
        if prob == "simple1": # using nesterov momentum
            alpha = .1222
            B = .55
            x_next, v_next = nesterov_momentum(g, x_last, v_last, alpha, B)
            x_last = x_next
            v_last = v_next
        elif prob == "simple2": # using nesterov momemntum
            alpha = .11925
            B = .65
            x_next, v_next = nesterov_momentum(g, x_last, v_last, alpha, B)
            x_last = x_next
            v_last = v_next
        elif prob == "simple3": # using momentum method
            alpha = .05
            B = .2
            x_next, v_next = momentum(g, x_last, v_last, alpha, B)
            x_last = x_next
            v_last = v_next
        elif prob == "secret1":
            alpha = .11925
            B = .65
            x_next, v_next = nesterov_momentum(g, x_last, v_last, alpha, B)
            x_last = x_next
            v_last = v_next
        elif prob == "secret2":
            alpha = .11925
            B = .65
            x_next, v_next = nesterov_momentum(g, x_last, v_last, alpha, B)
            x_last = x_next
            v_last = v_next
        else:
            return float("nan")
        

    x_best = x_last
    return x_best