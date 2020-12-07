# Command line Tic Tac Toe Game
import numpy as np
import pickle
#from Part1_TicTacToe import multiclass_MLP
from check_winner import check_winner, full_board
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings("ignore")


def drawBoard(board):
	# This function prints out the board that it was passed.
	# "board" is a list of 10 strings representing the board (ignore index 0)
	print('     |     |  ')
	print(' ' + str(board[0]) + '   | ' + str(board[1]) + '   | ' + str(board[2]))
	print('     |     |')
	print('-----------------')
	print('     |     |')
	print(' ' + str(board[3]) + '   | ' + str(board[4]) + '   | ' + str(board[5]))
	print('     |     |')
	print('-----------------')
	print('     |     |')
	print(' ' + str(board[6]) + '   | ' + str(board[7]) + '   | ' + str(board[8]))
	print('     |     |')
 
def if_exists(final_moves,ele):
	for i in final_moves:
		if ele==i:
			return 1
	return 0

def predict_ai_move(board, input_model):
	move_list = []

	if input_model==1: # Classifier
		multiclass_MLP_model = pickle.load(open("Saved_Models/multiclass_MLP.sav","rb"))
		ai_move = multiclass_MLP_model.predict([board])[0]

	if input_model==2: # MLP Regressor
		for i in range(0,9):
			class_model = pickle.load(open("Saved_Models/MLPRegressor"+str(i)+".sav","rb"))
			move = class_model.predict([board])[0]
			move_list.append(move)
		ai_move = np.argmax(move_list)

	if input_model==3: # KNN Regressor
		for i in range(0,9):
			class_model = pickle.load(open("Saved_Models/KNNRegressor"+str(i)+".sav","rb"))
			move = class_model.predict([board])[0]
			move_list.append(move)
		ai_move = np.argmax(move_list)

	if input_model == 4: # Linear Regressor
		for i in range(0,9):
			class_weights = np.load("Saved_Models/LinearRegressor_weights"+str(i)+".npy")
			move = np.dot(board,class_weights)
			move_list.append(move)

		ai_move = np.argmax(move_list)
	
	return ai_move, move_list



board = [0]*9

i = input("Would you like to use \nClassifier (1)\nMLP Regressor (2)\nKNN Regressor (3)\nLinear Regressor (4)\n? ")
if i == '1':
	input_model = 1
elif i == '2':
	input_model = 2
elif i == '3':
	input_model = 3
elif i == '4':
	input_model = 4
else:
	sys.exit('Enter 1, 2 or 3.')

while check_winner(board)==0 and full_board(board)==0:

	x_play = int(input("\nEnter the player X move\n"))
	
	while x_play<0 or x_play>8:
		print("\nInvalid Input. Try Again!\n")
		x_play = int(input("\nEnter the player X move\n"))
	if board[x_play]==0:
		board[x_play] = +1
	elif board[x_play]!=0:
		err = x_play
		while x_play==err:
			print("\nPosition already occupied! Try Again\n")
			x_play = int(input("\nEnter the player X move\n"))
			while x_play<0 or x_play>8:
				print("\nInvalid Input. Try Again!\n")
				x_play = int(input("\nEnter the player X move\n"))
			board[x_play] = +1
			
	drawBoard(board)
	if check_winner(board)==0 and full_board(board)==0:
		print("\nAI's turn\n")

		# Use various models to predict AI's next move
		ai_move, move_list = predict_ai_move(board, input_model)
		
		if board[ai_move]!=0:
			move_ind = np.argsort(move_list)
			
			move_ind = np.flip(move_ind)
			i=1
			while board[move_ind[i+1]]!=0:
				i=i+1
			ai_move = move_ind[i+1]
		#import pdb; pdb.set_trace()
		board[ai_move] = -1

		drawBoard(board)
	else:
		break	
		

flag=0
if check_winner(board) !=0:
	winner = check_winner(board)
	if winner==-1:
		print("_________    AI wins!!!    _________    ")
		flag=1
	else:
		print("_________    You win!!!    _________    ")
		flag=1

if full_board(board)!=0 and flag==0:
	print("_________    Draw Match!!!    _________    ")


