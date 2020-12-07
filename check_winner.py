import numpy as np

def checkRows(board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
    return 0

def checkDiagonals(board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
        return board[0][len(board)-1]
    return 0

def check_winner(board):
    board = np.array(board).reshape(3,3)
    #transposition to check rows, then columns
    for newBoard in [board, np.transpose(board)]:
        result = checkRows(newBoard)
        if result:
            return result
    return checkDiagonals(board)

def full_board(board):
    board = np.array(board).reshape(3,3)
    count=0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j]!=0:
                count= count+1
    if count==9:
        return 1
    else:
        return 0

