# Solve Every Sudoku Puzzle

# See http://norvig.com/sudoku.html

# Throughout this program we have:
# r is a row,    e.g. 'A'
# c is a column, e.g. '3'
# s is a square, e.g. 'A3'
# d is a digit,  e.g. '9'
# u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
# grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
# values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}


import numpy as np

class SudokuSolver:

    def __init__(self):
        self.digits = '123456789'
        self.rows = 'ABCDEFGHI'
        self.cols = self.digits
        self.squares = self.__cross(self.rows, self.cols)
        self.unitlist = ([self.__cross(self.rows, c) for c in self.cols] +
                    [self.__cross(r, self.cols) for r in self.rows] +
                    [self.__cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
        self.units = dict((s, [u for u in self.unitlist if s in u])
                    for s in self.squares)
        self.peers = dict((s, set(sum(self.units[s], []))-set([s]))
                    for s in self.squares)


    def __cross(self, A, B):
        "Cross product of elements in A and elements in B."
        return [a+b for a in A for b in B]


    ################ Parse a Grid ################
    def __parse_grid(self, grid):
        """Convert grid to a dict of possible values, {square: digits}, or
        return False if a contradiction is detected."""
        # To start, every square can be any digit; then assign values from the grid.
        values = dict((s, self.digits) for s in self.squares)
        for s, d in self.__grid_values(grid).items():
            if d in self.digits and not self.__assign(values, s, d):
                return False  # (Fail if we can't assign d to square s.)
        return values


    def __grid_values(self, grid):
        "Convert grid into a dict of {square: char} with '0' or '.' for empties."
        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))

    ################ Constraint Propagation ################
    def __assign(self, values, s, d):
        """Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return False if a contradiction is detected."""
        other_values = values[s].replace(d, '')
        if all(self.__eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return False


    def __eliminate(self, values, s, d):
        """Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected."""
        if d not in values[s]:
            return values  # Already eliminated
        values[s] = values[s].replace(d, '')
        # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
        if len(values[s]) == 0:
            return False  # Contradiction: removed last value
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self.__eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False
        # (2) If a unit u is reduced to only one place for a value d, then put it there.
        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False  # Contradiction: no place for this value
            elif len(dplaces) == 1:
                # d can only be in one place in unit; assign it there
                if not self.__assign(values, dplaces[0], d):
                    return False
        return values


    ################ Search ################
    def __solve(self, grid):
        return self.__search(self.__parse_grid(grid))


    def __search(self, values):
        "Using depth-first search and propagation, try all possible values."
        if values is False:
            return False  # Failed earlier
        if all(len(values[s]) == 1 for s in self.squares):
            return values  # Solved!
        # Chose the unfilled square s with the fewest possibilities
        n, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        return self.__some(self.__search(self.__assign(values.copy(), s, d))
                    for d in values[s])

    ################ Utilities ################
    def __some(self, seq):
        "Return some element of seq that is true."
        for e in seq:
            if e:
                return e
        return False

    def __convert_to_string(self, grid_array):

        grid_string = ""

        for row in grid_array:
            for digit in row:
                grid_string += str(digit)

        return grid_string

    def __convert_to_array(self, values):

        grid_array = []

        print(values)

        for square in values:
            grid_array.append(int(values[square]))

        np_grid = np.array(grid_array)

        return np.reshape(np_grid, (9, 9))



    ################ System test ################
    def __solved(self, values):
        "A puzzle is solved if each unit is a permutation of the digits 1 to 9."
        def unitsolved(unit):
            return set(values[s] for s in unit) == set(self.digits)
        return values is not False and all(unitsolved(unit) for unit in self.unitlist)

    def solve_array(self, grid_array):
        """Attempt to solve grid."""

        grid = self.__convert_to_string(grid_array)
        values = self.__solve(grid)

        if not self.__solved(values):
            return False

        return self.__convert_to_array(values)
