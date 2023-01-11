# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
from matplotlib import pyplot as plt
import time


# Problem 1
def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client()
    dview = client[:]
    dview.execute("import scipy.sparse as sparse")

    # Don't forget to close
    client.close()
    
    return dview

# print(initialize())

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # Initialize client and direct view
    client = Client()
    dview = client[:]

    # Block push and pull
    dview.block = True
    dview.push(dx)
    dview.pull('a')

    # Check
    keys = dx.keys()
    for key in keys:
        assert dview.pull(key) == [dx[key]]*8

    # Don't forget to close
    client.close()


# variables({'a':10, 'b':5})


# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # Initialize client and dview
    client = Client()
    dview = client[:]
    dview.execute("import numpy as np")

    # Define draw function from normal
    def draw(n):
        x = np.random.normal(0,1,n)
        means = np.mean(x)
        maxs = np.max(x)
        mins = np.min(x)

        return [means, maxs, mins]

    # Parallelize
    results = dview.map_sync(draw, [n]*len(client.ids))
    results = np.transpose(results)

    client.close()
    # return results
    return results[0], results[2], results[1]


# print(prob3())


# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    ns = [1000000, 5000000, 10000000, 15000000]

    # Define draw function again
    def draw(n):
        x = np.random.normal(0,1,n)
        means = np.mean(x)
        maxs = np.max(x)
        mins = np.min(x)

        return [means, maxs, mins]

    fast = []
    times = []

    # Iterate through number of draws
    for n in ns:
        # Time problem 3
        start = time.time()
        prob3(n)
        fast.append(time.time()-start)

        # Time same thing serially
        means, maxs, mins = [], [], []
        start = time.time()
        for i in range(8):
            results = draw(n)
            means.append(results[0])
            maxs.append(results[0])
            mins.append(results[0])
        end = time.time()
        times.append(end-start)

    # Plot it
    plt.plot(ns, times, label="Serially")
    plt.plot(ns, fast, label="Parallel")
    plt.ylabel("Time")
    plt.xlabel("Number of Draws")
    plt.legend()
    plt.show()

# prob4()


# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # Initialize
    client = Client()
    dview = client[:]
    dview.block = True

    # Create linspace and find h
    X = np.linspace(a,b,n)
    h = X[1] - X[0]

    # Define area function
    def area(k, l):
        return f(k) + f(l)

    # Parallelize and close
    results = dview.map(area, X[:-1], X[1:])
    client.close()

    return np.sum(results)*h/2


# f = lambda x: x**2
# estimate = parallel_trapezoidal_rule(f, 0, 2)
# print(estimate)


