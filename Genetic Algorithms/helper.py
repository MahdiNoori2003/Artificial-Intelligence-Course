from sympy import symbols, Poly 
import random
import matplotlib.pyplot as plt
import numpy as np

def calculate_mse(predictions, targets):
    """
    Calculates the Mean Squared Error (MSE) between the predicted values and the target values.

    Args:
        predictions: List or NumPy array of predicted values.
        targets: List or NumPy array of target values.

    Returns:
        The MSE value.
    """
    squared_errors = [(predict - target) ** 2 for predict, target in zip(predictions, targets)]
    mse = sum(squared_errors) / len(predictions)
    return mse

def test_gen(coeffs, num_of_tests,boundries=[-20,20]):
    """
    Generates a set of test cases for evaluating a polynomial function.

    Args:
        coeffs (list): Coefficients of the polynomial function in descending order of degree.
        num_of_tests (int): Number of test cases to generate.

    Returns:
        list: A list of tuples representing the generated test cases. Each tuple contains an x-value and its corresponding y-value.
    """
    tests = []
    for _ in range(num_of_tests):
        x = random.randrange(boundries[0], boundries[1])
        y = 0
        for j in range(len(coeffs)):
            y += coeffs[j] * (x ** j)
        tests.append((x, y))
    return tests


def curve_plot(coeffs, points, title='Fitted Curve'):
    """
    Plots a curve using the coefficients of a polynomial and a set of points.

    Args:
        coeffs (list): Coefficients of the polynomial function in descending order of degree.
        points (list): List of tuples representing the points to be plotted. Each tuple contains an x-value and its corresponding y-value.
        title (str, optional): Title of the plot. Defaults to 'Fitted Curve'.

    Returns:
        None
    """
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]

    x = np.linspace(min(x_points), max(x_points), 100)
    y = np.polyval(coeffs[::-1], x)

    plt.plot(x, y,label="Fitted Curve")
    plt.scatter(x_points, y_points, color='red',label="Points")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def create_equation(coeffs):
    """
    Creates an equation string based on the given coefficients.

    Args:
        coeffs (list): Coefficients of the polynomial equation in descending order of degree.

    Returns:
        str: The equation string.
    """
    x = symbols('x')
    poly = Poly(coeffs, x)
    equation = poly.as_expr()

    return str(equation.__repr__())
