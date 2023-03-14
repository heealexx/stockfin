import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.depth_setting = 5

    def game_completed(self, board, player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))
        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False
        def check_verticle(b):
            return check_horizontal(b.T)
        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def update_board(self, board, move, player_num):
        if 0 in board[:,move]:
            update_row = -1
            for row in range(1, board.shape[0]):
                update_row = -1
                if board[row, move] > 0 and board[row-1, move] == 0:
                    update_row = row-1
                elif row==board.shape[0]-1 and board[row, move] == 0:
                    update_row = row

                if update_row >= 0:
                    board[update_row, move] = player_num
                    break

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithmC

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        maxplayer = self.player_number
        minplayer = 0
        if maxplayer == 1:
            minplayer = 2
        else: 
            minplayer = 1

        def update_board(board, move, player_num):
            if 0 in board[:,move]:
                update_row = -1
                for row in range(1, board.shape[0]):
                    update_row = -1
                    if board[row, move] > 0 and board[row-1, move] == 0:
                        update_row = row-1
                    elif row==board.shape[0]-1 and board[row, move] == 0:
                        update_row = row

                    if update_row >= 0:
                        board[update_row, move] = player_num
                        break

        def minimax(board, depth, isMaximizing, alpha, beta):
            """
            Minimax with alpha pruning algo
            """
            if depth == 0:
                return self.evaluation_function(board)
            if isMaximizing:
                value = -1000
                for i in range(0, 7):
                    board_copy = board.copy()
                    self.update_board(board_copy, i, maxplayer)
                    if (self.game_completed(board_copy, maxplayer)):
                        value = 1000000
                    else:
                        value = max(minimax(board_copy, depth-1, False, alpha, beta), value)
                    if value >= beta:
                        break
                    alpha = max(alpha, value)
                return value
            else:
                value = 1000
                for j in range(0, 7):
                    board_copy = board.copy()
                    self.update_board(board_copy, j, minplayer)
                    if (self.game_completed(board_copy, minplayer)):
                        value = -1000000
                    else:
                        value = min(minimax(board_copy, depth-1, True, alpha, beta), value)
                    if value <= alpha:
                        break
                    beta = min(beta, value)
                return value

        vals = list()
        for i in range(0, 7):
            board_copy = board.copy()
            self.update_board(board_copy, i, maxplayer)
            if (self.game_completed(board_copy, maxplayer)):
                vals.append(1000000)
            else:
                vals.append(minimax(board_copy, self.depth_setting, False, -1000, 1000))
        vals_max = max(vals)
        print(vals)
        for i in range(0, 7):
            if vals[i] == vals_max:
                return i

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        max_player = self.player_number
        rand_player = 0
        if max_player == 1:
            rand_player = 2
        else:
            rand_player = 1

        def expectimax(board, depth, isMaximizing):
            if depth == 0:
                return self.evaluation_function(board)
            if isMaximizing:
                a = -10000
                for i in range(0, 7):
                    board_copy = board.copy()
                    self.update_board(board_copy, i, max_player)
                    if (self.game_completed(board_copy, max_player)):
                        a = 1000000
                    else:
                        a = max(a, expectimax(board_copy, depth-1, False))
            else:
                a = 0
                for i in range(0, 7):
                    board_copy = board.copy()
                    self.update_board(board_copy, i, rand_player)
                    if (self.game_completed(board_copy, rand_player)):
                        a = -1000000
                    else:
                        a += ((expectimax(board_copy, depth-1, True)) / 7)

            return a

        vals = list()
        for i in range(0, 7):
            board_copy = board.copy()
            self.update_board(board_copy, i, max_player)
            if (self.game_completed(board_copy, max_player)):
                vals.append(1000000)
            else:
                vals.append(expectimax(board_copy, 4, False))
        vals_max = max(vals)
        for i in range(0, 7):
            if vals[i] == vals_max:
                return i

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        def check_horizontal(board, player_num):
            board_copy = board.copy()
            board_copy[board_copy == 0] = player_num
            pattern = np.full(4, player_num)
            total = 0
            for row in board_copy:
                for i in range(0, int((len(row)/2)+1)):
                    if (np.array_equal(pattern, row[i:i+4])):
                        total += 1
            return total

        def check_vertical(board, player_num):
            return check_horizontal(board.T, player_num)

        def check_diagonal(board, player_num):
            pattern = np.full(4, player_num)
            total = 0
            board_copy = board.copy()
            board_copy[board_copy == 0] = player_num
            for op in [None, np.fliplr]:
                op_board = op(board_copy) if op else board_copy
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                for i in range(0, 3):
                    if (np.array_equal(pattern, root_diag[i:i+4])):
                        total += 1

                for i in range(1, board_copy.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if len(diag) >= 4:
                            for i in range(0, int(len(diag)/2)):
                                if (np.array_equal(pattern, diag[i:i+4])):
                                    total += 1

            return total

        to_str = lambda a: ''.join(a.astype(str))

        player_num = self.player_number
        player_score = 0
        player_score += check_horizontal(board, player_num)
        player_score += check_vertical(board, player_num)
        player_score += check_diagonal(board, player_num)

        opp_num = 0
        if player_num == 1:
            opp_num = 2
        else:
            opp_num = 1
        opp_score = 0
        opp_score += check_horizontal(board, opp_num)
        opp_score += check_vertical(board, opp_num)
        opp_score += check_diagonal(board, opp_num)

        return player_score - opp_score


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

