import random
import math
import time
from BaseAI import BaseAI

MAX_DEPTH = 1  # Adjust the maximum depth based on experimentation
MAX_TIME = 0.2  # Time limit for each move


class IntelligentAgent(BaseAI):
    def __init__(self, max_depth=MAX_DEPTH, max_time=MAX_TIME):
        self.max_depth = max_depth
        self.max_time = max_time

    def evaluate_board(self, grid):
        nonempty = 16 - len(grid.getAvailableCells())
        return self.heuristic_monotone(grid) + self.heuristic_smoothness(grid) - (nonempty ** 2)

    def heuristic_smoothness(self, grid):
        smoothness = 0
        for x in range(4):
            for y in range(4):
                current_val = grid.map[x][y] or 2
            
                neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
            
                valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < 4 and 0 <= ny < 4]
            
                differences = [abs(current_val - (grid.map[nx][ny] or 2)) for nx, ny in valid_neighbors]
                if differences:
                    smoothness -= min(differences)
        return smoothness
    
    def heuristic_monotone(self, grid):
        def score_line(line):
            score = 0
            for i in range(len(line) - 1):
                if line[i] > line[i + 1]:
                    score += line[i] - line[i + 1]
                else:
                    score -= line[i + 1] - line[i]
            return score

        total_score = 0
        for x in range(4):
            total_score += score_line(grid.map[x])  # Rows
            total_score += score_line([grid.map[y][x] for y in range(4)])  # Columns

        return total_score


    def getMove(self, grid):
        """ Returns a randomly selected cell if possible """
        self.start_time = time.time()
        best_move = None
        max_eval = -math.inf
        alpha = -math.inf
        beta = math.inf

        # moves = sorted(grid.getAvailableMoves(), key=lambda x: self.evaluate_board(x[1]), reverse=True)
        moves = grid.getAvailableMoves()
        for move, new_grid in moves:
            eval = self.expectiminimax_alpha_beta(new_grid, self.max_depth, 1.0, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            if max_eval >= beta:
                break
            if max_eval > alpha:
                alpha = max_eval
                
        if best_move == None:
            best_move = random.choice(moves)[0]
        return best_move

    def expectiminimax_alpha_beta(self, grid, depth, prob_node, alpha, beta):
        if depth <= 0 or not grid.canMove():
            return self.evaluate_board(grid)

        total_eval = 0
        total_probability = 0
        available_cells = grid.getAvailableCells()

        for cell in available_cells:
            for probability, tile_value in ((0.1, 4), (0.9, 2)):
                prob_node1 = probability * prob_node
                if 0.9 * prob_node1 < 0.1 and len(available_cells) > 4:
                    continue
                new_grid = grid.clone()
                new_grid.insertTile(cell, tile_value)
                total_eval += probability * self.maximize(new_grid, depth, prob_node1, alpha, beta)
                total_probability += probability
            if(time.time() >= self.start_time + self.max_time):
                break
        if total_probability == 0:
            return self.evaluate_board(grid)

        return total_eval / total_probability

    def maximize(self, grid, depth, prob_of_node, alpha, beta):
        max_eval = -math.inf
        for move, new_grid in grid.getAvailableMoves():
            eval = self.expectiminimax_alpha_beta(new_grid, depth - 1, prob_of_node, alpha, beta)
            if eval > max_eval:
                max_eval = eval
            if max_eval >= beta:
                break
            if max_eval > alpha:
                alpha = max_eval
        return max_eval
