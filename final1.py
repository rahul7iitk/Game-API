
import copy
import pickle
import time


import tensorflow as tf
import numpy as np
import copy
import random
import math
import pickle
import matplotlib.pyplot as plt


input_nodes=100
hidden_nodes=100
output_nodes=100
gamma=0.5

def print_board(s,board):

	# WARNING: This function was crafted with a lot of attention. Please be aware that any
	#          modifications to this function will result in a poor output of the board 
	#          layout. You have been warn. 

	#find out if you are printing the computer or user board
	player = "Computer"
	if s == "u":
		player = "User"
	
	print "The " + player + "'s board look like this: \n"

	#print the horizontal numbers
	print " ",
	for i in range(10):
		print "  " + str(i+1) + "  ",
	print "\n"

	for i in range(10):
	
		#print the vertical line number
		if i != 9: 
			print str(i+1) + "  ",
		else:
			print str(i+1) + " ",

		#print the board values, and cell dividers
		for j in range(10):
			if board[i][j] == -1:
				print ' ',	
			elif s == "u":
				print board[i][j],
			elif s == "c":
				if board[i][j] == "*" or board[i][j] == "$":
					print board[i][j],
				else:
					print " ",
			
			if j != 9:
				print " | ",
		print
		
		#print a horizontal line
		if i != 9:
			print "   ----------------------------------------------------------"
		else: 
			print 




def place_ship(board,ship,s,ori,x,y):

	#place ship based on orientation
	if ori == "v":
		for i in range(ship):
			board[x+i][y] = s
	elif ori == "h":
		for i in range(ship):
			board[x][y+i] = s

	return board	

def validate(board,ship,x,y,ori):

	#validate the ship can be placed at given coordinates
	if ori == "v" and x+ship > 10:
		return False
	elif ori == "h" and y+ship > 10:
		return False
	else:
		if ori == "v":
			for i in range(ship):
				if board[x+i][y] != -1:
					return False
		elif ori == "h":
			for i in range(ship):
				if board[x][y+i] != -1:
					return False
		
	return True

def v_or_h():

	#get ship orientation from user
	while(True):
		user_input = raw_input("vertical or horizontal (v,h) ? ")
		if user_input == "v" or user_input == "h":
			return user_input
		else:
			print "Invalid input. Please only enter v or h"

def get_coor():
	
	while (True):
		user_input = raw_input("Please enter coordinates (row,col) ? ")
		try:
			#see that user entered 2 values seprated by comma
			coor = user_input.split(",")
			if len(coor) != 2:
				raise Exception("Invalid entry, too few/many coordinates.");

			#check that 2 values are integers
			coor[0] = int(coor[0])-1
			coor[1] = int(coor[1])-1

			#check that values of integers are between 1 and 10 for both coordinates
			if coor[0] > 9 or coor[0] < 0 or coor[1] > 9 or coor[1] < 0:
				raise Exception("Invalid entry. Please use values between 1 to 10 only.")

			#if everything is ok, return coordinates
			return coor
		
		except ValueError:
			print "Invalid entry. Please enter only numeric values for coordinates"
		except Exception as e:
			print e

def make_move(board,x,y):
	
	#make a move on the board and return the result, hit, miss or try again for repeat hit
	
	if board[x][y] == -1:
		return "miss"
	elif board[x][y] == '*' or board[x][y] == '$':
		return "try again"
	else:
		return "hit"



def user_move(board):
	
	#get coordinates from the user and try to make move
	#if move is a hit, check ship sunk and win condition
	while(True):
		#x,y = get_coor()
		x = random.randint(0,9)
		y = random.randint(0,9)
		res = make_move(board,x,y)
		if res == "hit":
			print "Hit at " + str(x+1) + "," + str(y+1)
			check_sink(board,x,y)
			board[x][y] = '$'
			
			if check_win(board):
				return "WIN"
			board = user_move(board)	
		elif res == "miss":
			print "Sorry, " + str(x+1) + "," + str(y+1) + " is a miss."
			board[x][y] = "*"
		elif res == "try again":
			print "Sorry, that coordinate was already hit. Please try again"

		if res != "try again":
			return board


def computer_move(board,x,y):
	
	#generate user coordinates from the user and try to make move
	#if move is a hit, check ship sunk and win condition
	while(True):
		
		res = make_move(board,x,y)
		if res == "hit":
			print "Hit at " + str(x+1) + "," + str(y+1)
			check_sink(board,x,y)
			board[x][y] = '$'
			if check_win(board):
				return "WIN"

			global current_state,actions

			current_state[0,10*x+y]= 1*(10*x+y in ship_positions)
			#drawRadar(current_state)
			actions.append(10*x+y)

			
			probs = session.run(probabilities,feed_dict={input_layer:current_state})
			
			prob = [p * (index not in actions) for index, p in enumerate(probs[0])]
			prob = [p / sum(prob) for p in prob]
			current_action = np.argmax(probs)
			board = computer_move(board,current_action/10,current_action%10)
			

		elif res == "miss":
			print "Sorry, " + str(x+1) + "," + str(y+1) + " is a miss."
			board[x][y] = "*"

		if res != "try again":
			
			return board


def user_place_ships(board,ships):

	for ship in ships.keys():

		#get coordinates from user and vlidate the postion
		valid = False
		while(not valid):

			print_board("u",board)
			print "Placing a/an " + ship
			x,y = get_coor()
			ori = v_or_h()
			valid = validate(board,ships[ship],x,y,ori)
			if not valid:
				print "Cannot place a ship there.\nPlease take a look at the board and try again."
				raw_input("Hit ENTER to continue")

		#place the ship
		board = place_ship(board,ships[ship],ship[0],ori,x,y)
		print_board("u",board)
		
	raw_input("Done placing user ships. Hit ENTER to continue")
	return board


def computer_place_ships(board,ships):
    for ship in ships.keys():
        valid=False
        while(not valid):
            x=random.randint(0,9)
            y=random.randint(0,9)
            o=random.randint(0,1)
            if(o==0):
                ori="v"
            else:
                ori="h"
            valid=validate(board,ships[ship],x,y,ori)
        board=place_ship(board,ships[ship],ship[0],ori,x,y)
    return board

def place_ship(board,ship,s,ori,x,y):
    #place ship based on orientation
    if ori == "v":
        for i in range(ship):
            board[x+i][y] = s
    elif ori == "h":
        for i in range(ship):
            board[x][y+i] = s
    return board


def validate(board,ship,x,y,ori):
    #validate the ship can be placed at given coordinates
    if ori == "v" and x+ship > 10:
        return False
    elif ori == "h" and y+ship > 10:
        return False
    else:
        if ori == "v":
            for i in range(ship):
                if board[x+i][y] != -1:
                    return False
        elif ori == "h":
            for i in range(ship):
                if board[x][y+i] != -1:
                    return False

    return True

def findships(board):
    ship_pos=[]
    for i in range(0,10):
        for j in range(0,10):
            if(board[i][j]!=-1):
                ship_pos.append(10*(i)+j)
    return ship_pos

def funcdist(a,b):
    x1 = a/10
    y1 = a%10

    x2 = b/10
    y2 = b%10

    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def check_sink(board,x,y):

	#figure out what ship was hit
	if board[x][y] == "A":
		ship = "Aircraft Carrier"
	elif board[x][y] == "B":
		ship = "Battleship"
	elif board[x][y] == "S":
		ship = "Submarine" 
	elif board[x][y] == "D":
		ship = "Destroyer"
	elif board[x][y] == "P": 
		ship = "Patrol Boat"
	
	#mark cell as hit and check if sunk
	board[-1][ship] -= 1
	if board[-1][ship] == 0:
		print ship + " Sunk"
		

def check_win(board):
	
	#simple for loop to check all cells in 2d board
	#if any cell contains a char that is not a hit or a miss return false
	for i in range(10):
		for j in range(10):
			if board[i][j] != -1 and board[i][j] != '*' and board[i][j] != '$':
				return False
	return True


def rewards_calculator(hitlog,actions):
    dist=[1]
    for i in range(1,len(actions)):
        dist.append(funcdist(actions[i],actions[i-1]))


    hit_log_weighted = [(item -float(17 - sum(hitlog[:index])) / float(100 - index)) * (gamma ** index) for index, item in enumerate(hitlog)]
    hit_log_weighted_1 = []
    for i in range(len(hit_log_weighted)):
        hit_log_weighted_1.append(hit_log_weighted[i]/dist[i])
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hitlog))]

with open('gameaiweightsnewdist.pickle','rb') as f:
    dict=pickle.load(f)
    w1=dict['weights1']
    w2=dict['weights2']
    b1=dict['baises1']
    b2=dict['baises2']

#value function approximator(neural network)
input_layer=tf.placeholder(tf.float32,[1,input_nodes])
labels=tf.placeholder(tf.int64)
learning_rate=tf.placeholder(tf.float32,shape=[])
'''
weights1=tf.Variable(tf.truncated_normal([input_nodes,hidden_nodes],stddev=0.1/np.sqrt(100.0)))
baises1=tf.Variable(tf.zeros(shape=(1,hidden_nodes)))
weights2=tf.Variable(tf.truncated_normal([hidden_nodes,output_nodes],stddev=0.1/np.sqrt(100.0)))
baises2 =tf.Variable(tf.zeros(shape=(1,output_nodes)))
'''

weights1=tf.Variable(w1)
baises1=tf.Variable(b1)
weights2=tf.Variable(w2)
baises2 =tf.Variable(b2)

a1=tf.tanh(tf.matmul(input_layer,weights1)+baises1)

z2=tf.matmul(a1,weights2)+baises2
probabilities=tf.nn.softmax(z2)

cost=(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=z2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

game_progress=[]
init=tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	game_lengths = []
	game_lengths_avg = []
	x1 = []
	arr=[]


	for game in range(1):
		print game
		states=[]
		actions=[]
		current_state=np.zeros(shape=(1,100),dtype=np.float32)
		current_state[0,:]=-1.0
		hitlog=[]
		ships = {"Aircraft Carrier":5, "Battleship":4, "Submarine":3,"Destroyer":3,"Patrol Boat":2}
		board = []
		for i in range(10):
			board_row = []
			for j in range(10):
				board_row.append(-1)
			board.append(board_row)
		board.append(copy.deepcopy(ships))
		user_board = copy.deepcopy(board)
		user_board = computer_place_ships(user_board,ships)
		board = user_place_ships(board,ships)
		ship_positions=findships(board)
		flag=0
		#time.sleep(1.0/1)
		while(1):
			#DISPLAYSURF.fill(bgcolor)
			
			#time.sleep(1.0)
			states.append(copy.deepcopy(current_state))
			
			#current_action = np.random.choice(100,p=prob)


			print_board("c",user_board)
			user_board = user_move(user_board)

			if (user_board=="WIN"):
				print "USER WON"
				quit()


			print_board("c",user_board)

			#raw_input("To end user turn hit ENTER")



			



			while(True):
		

				probs = session.run(probabilities,feed_dict={input_layer:current_state})
			
				prob = [p * (index not in actions) for index, p in enumerate(probs[0])]
				prob = [p / sum(prob) for p in prob]
				current_action = np.random.choice(100,p=prob)
				print current_action

				res = make_move(board,current_action/10,current_action%10)
				if res == "hit":
					print "Hit at " + str(current_action/10+1) + "," + str(current_action%10+1)
					check_sink(board,current_action/10,current_action%10)
					board[current_action/10][current_action%10] = '$'
					if check_win(board):
						board = "WIN"
						print "AI WON"
						quit()
					print_board("u",board)
					current_state[0,current_action]= 1*(current_action in ship_positions)
					actions.append(current_action)

			
					continue
			

				elif res == "miss":
					print "Sorry, " + str(current_action/10+1) + "," + str(current_action%10+1) + " is a miss."
					board[current_action/10][current_action%10] = "*"
					current_state[0,current_action]= 1*(current_action in ship_positions)
					
					actions.append(current_action)

				if res != "try again":
			
					break




			if(board=="WIN"):
				print "AI WON"
				quit()

			print current_action/10,current_action%10
			print_board("u",board)
			




