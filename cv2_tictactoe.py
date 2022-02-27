
#THE INVINCIBLE TIC-TAC-TOE



import numpy as np
import random
import cv2
import numpy as np
import math
import time

#Declare the game board
def arena():
    return(np.array([' ' for x in range(10)]))

#Print the game board
def spawn_board(board):
    print("  " + board[1] + " | "+board[2] + " | "+board[3] +"          | 1 | 2 | 3 |")
    print(" --- --- ---          --- --- --- ")
    print("  " + board[4] + " | "+board[5] + " | "+board[6] +"   <==>   | 4 | 5 | 6 |")
    print(" --- --- ---          --- --- --- ")
    print("  " + board[7] + " | "+board[8] + " | "+board[9] +"          | 7 | 8 | 9 |")
    print("\n-----------------------------------\n ")

#To Check if given location is already occupied
def valid_loc(board, loc):
    if board[loc]==' ':
        return(True)
    else:
        return(False)

#Move by human player
def user_move(board,loc,player):
    board[loc] = player

#To check winning combinations
def endgame(board, player):
    #Horizontal win
    if (board[1]== player and board[2]== player and board[3]== player):
        return(True)
    if (board[4]== player and board[5]== player and board[6]== player):
        return(True)
    if (board[7]== player and board[8]== player and board[9]== player):
        return(True)

    #Vertical win
    if (board[1]== player and board[4]== player and board[7]== player):
        return(True)
    if (board[2]== player and board[5]== player and board[8]== player):
        return(True)
    if (board[3]== player and board[6]== player and board[9]== player):
        return(True)

    #Diagonal win
    if (board[1]== player and board[5]== player and board[9]== player):
        return(True)
    if (board[3]== player and board[5]== player and board[7]== player):
        return(True)

# To check if game is tie
def is_tie(board):
    l=0
    for i in range(1,10):
        if board[i]!=' ':
            l = l+1
    if l==9:
        return(True)
    else:
        return(False)

# To return the value of each possible state of the board
# If player wins value is 10
# if opponent wins the value is -10
# Value is 0 in all other cases 
def t_states(board, p, opp):
    value = 0
    #Vertical win
    if board[1]==p and board[2]==p and board[3]==p:
        value = 10        
    if board[4]==p and board[5]==p and board[6]==p:
        value = 10
    if board[7]==p and board[8]==p and board[9]==p:
        value = 10
    
    if board[1]==opp and board[2]==opp and board[3]==opp:
        value = -10
    if board[4]==opp and board[5]==opp and board[6]==opp:
        value = -10
    if board[7]==opp and board[8]==opp and board[9]==opp:
        value = -10

    #Horizontal Win
    if board[1]==p and board[4]==p and board[7]==p:
        value = 10        
    if board[2]==p and board[5]==p and board[8]==p:
        value = 10
    if board[3]==p and board[6]==p and board[9]==p:
        value = 10

    if board[1]==opp and board[4]==opp and board[7]==opp:
        value = -10
    if board[2]==opp and board[5]==opp and board[8]==opp:
        value = -10
    if board[3]==opp and board[6]==opp and board[9]==opp:
        value = -10

    #Diagonal WIn
    if board[1]==p and board[5]==p and board[9]==p:
        value = 10        
    if board[3]==p and board[5]==p and board[7]==p:
        value = 10

    if board[1]==opp and board[5]==opp and board[9]==opp:
        value = -10
    if board[3]==opp and board[5]==opp and board[7]==opp:
        value = -10

    return(value)

# THE MINIMAX FUNCTION
# This function compares all cases to get best possible score/value
# Depth is the position or distance of the game board from initial(empty) board
# ismax is boolean value to check if the player is max or min
def minimax(board, depth, ismax, p1, opp1):
    value = t_states(board, p1, opp1)

    #Max wins
    if (value == 10):
        return (value - depth) # depth is added to decrease score
        # Computer will take shorter games paths
    #Min wins
    if (value == -10):
        return (value + depth) # depth is added to decrease score(magnitude)
        # Computer will take shorter game paths
        # Computer will therefore take faster wins

    if (is_tie(board)==True):
        return 0

    #Max makes move
    if (ismax == True):
        best_score = -1000

        for i in range(1,10):
            if board[i]==' ':

                #make move
                board[i] = p1

                #minimax recursion
                curr_score = minimax(board, depth+1, False, p1, opp1)
                # Max player will choose score with highest value
                best_score = max(best_score , curr_score)

                #return board to original state
                board[i] = ' '
        return best_score
    #Min makes move
    else:
        best_score = 1000

        for i in range(1,10):
            if board[i]==' ':

                #make move
                board[i] = opp1

                #minimax recursion
                curr_score = minimax(board, depth+1, True, p1, opp1)
                # Min player will choose score with lowest value
                best_score = min(best_score , curr_score)

                #return board to original state
                board[i] = ' '
        return best_score
    

# This function will choose the best move -
# - On the basis of score/value provided by minimax function
def best_move(board, p, opp):
    best_score = -1000
    #bestmove = 0
    for i in range(1,10):
        if board[i] == ' ':
            board[i] = p
            score = minimax(board, 0, False, p, opp)
            board[i] = ' '
            if score > best_score:
                best_score = score
                bestmove = i
    return bestmove


# To mark the moves on the virtual game board
def mark(img_board,pos,p):
    if pos == 1:
        img_board = cv2.putText(img_board,p,(4,160), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 2:
        img_board = cv2.putText(img_board,p,(174,160), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 3:
        img_board = cv2.putText(img_board,p,(344,160), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 4:
        img_board = cv2.putText(img_board,p,(4,320), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 5:
        img_board = cv2.putText(img_board,p,(174,320), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 6:
        img_board = cv2.putText(img_board,p,(344,320), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 7:
        img_board = cv2.putText(img_board,p,(4,480), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 8:
        img_board = cv2.putText(img_board,p,(174,480), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    elif pos == 9:
        img_board = cv2.putText(img_board,p,(344,480), cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),3,cv2.LINE_AA)
    
    return img_board
 
# This Function respective returns values for respective game modes
# def get_game_mode():
#     while True:
#         game_mode = int(input("Enter your selection no. : "))
#         if game_mode in range(1,4):
#             break
#         else:
#             print("Invalid!! Please select correct values!")
#     # The game value returns [comp, human, turn]
#     # X will always play first(I made this rule for the sake of convenience)
#     if game_mode == 1:
#         game_values = ['O','X',1] # Human First
    
#     elif game_mode == 2:
#         game_values = ['X','O',0] # Computer First
    
#     elif game_mode == 3:
#         guess = random.randrange(1,11)
#         if guess%2 == 0:
#             game_values = ['O','X',1] # if random num even then Human first
#         elif guess%2 == 1:
#             game_values = ['X','O',0] # if random num odd then Comp first
#     return game_values
    
    

# Invincible Tic tac toe - main function


print("    Welcome to the INVINCIBLE TIC-TAC-TOE ,Challenger\n")
print("     Please select the game mode : \n")
print("  1. Human first\n  2. Computer first\n  3. Random\n")

#mode = get_game_mode()

board = arena()

spawn_board(board)

# p1 = mode[0]
# opp1 = mode[1]
p1 = 'O'
opp1 = 'X'

print("!! Refer to the table provided to make your moves !!\n")

entry = 0
val = -10
data = []
flag = 0
result = 0

#turn = mode[2]
turn = 1
if turn == 1:
    print("\nChallenger make the first move")
elif turn == 0:
    print("\nComputer will make the first move")


#---------------------------------------------------
# Game Board
img = np.zeros((512,512,3), np.uint8)
img = cv2.line(img,(170,0),(170,511),(255,0,0),5)
img = cv2.line(img,(340,0),(340,511),(255,0,0),5)
img = cv2.line(img,(0,170),(511,170),(255,0,0),5)
img = cv2.line(img,(0,340),(511,340),(255,0,0),5)
#---------------------------------------------------
cap = cv2.VideoCapture(0)
game_over = False
while game_over == False:
    if is_tie(board):
        print("!! The game is draw !! \n You are Good but not the best hehehe (;")
        game_over = True
    else:
        # Take each frame
        _, frame = cap.read()

        # region of interest
        d_reg = frame[100:400,100:400]

        cv2.rectangle(frame,(100,100),(400,400),(0,255,0),1)
        
        
        # # Convert BGR to HSV
        hsv_image = cv2.cvtColor(d_reg, cv2.COLOR_BGR2YCR_CB)
        # convert to gray
        gray = cv2.cvtColor(d_reg, cv2.COLOR_BGR2GRAY)
        # Thresholding
        ret , threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        
        # Dilation + erosion
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(threshold,kernel,iterations = 4)
        mask = cv2.erode(mask,kernel,iterations = 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        

        # median Blur
        mask = cv2.medianBlur(mask,5)
        
        
        # detecting hand by skin color
        lower_skin = np.array([0,58,30])
        upper_skin = np.array([33,255,255])
        filter_ = cv2.inRange(hsv_image, lower_skin, upper_skin)
        filter_ = cv2.erode(filter_,kernel,iterations = 1)
        filter_ = cv2.bitwise_not(filter_)
        
        # Finding combination of filter and mask
        comb = cv2.bitwise_or(filter_,mask)

        # cv2.imshow('comb', comb)


        # Finding Contours
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        

        # Convex Hull
        if contours == []:
            pass
        else:
            # Finding max Contour
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            
            # Finding solidity
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            #print(solidity)

            # Finding extent
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            rect_area = w*h
            extent = float(area)/rect_area
            #print(extent)

            # Finding Aspect ratio
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            #print(aspect_ratio)

            #print(solidity,'  ', extent, '  ', aspect_ratio)
            
            # Drawing Contours
            cv2.drawContours(d_reg, [hull], 0, (0,255,0), 2)
            
            # hull for finding defects
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            count_defects = 0
            count_defects2 = 0

            # Defects
            try:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                    d=(2*ar)/a
                    
                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                    
                
                    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                    if angle <= 90 and d>30:
                        count_defects += 1
                        cv2.circle(d_reg, far, 3, [255,0,0], -1)
                    
                    
                    # Finding angles greater than 90
                    if angle >= 90 and d>30:
                        count_defects2 += 1
                        cv2.circle(d_reg, far, 3, [0,0,255], -1)
            except:
                print("no defect")


        # Detecting sign on basis of ratios obtained
        try:
            if count_defects == 0 :
                if solidity > 0.94:
                    cv2.putText(frame,'9',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                    val = 9
                    entry += 1

                elif solidity > 0.85 and solidity < 0.94:
                    cv2.putText(frame,'0',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)

                elif solidity < 0.85:
                    if aspect_ratio > 0.68:
                        cv2.putText(frame,'6',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                        val = 6
                        entry += 1
                    else:
                        cv2.putText(frame,'1',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                        val = 1
                        entry += 1

            if count_defects == 1:
                if extent > 0.48:
                    cv2.putText(frame,'2',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                    val = 2
                    entry += 1

                elif extent < 0.43 or solidity > 0.65 or aspect_ratio > 0.71:
                    cv2.putText(frame,'7',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                    val = 7
                    entry += 1

            if count_defects == 2:
                if solidity < 0.65 or aspect_ratio > 0.70:
                    cv2.putText(frame,'8',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                    val = 8
                    entry += 1

                elif solidity > 0.66:
                    cv2.putText(frame,'3',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                    val = 3
                    entry += 1
            if count_defects == 3:
                cv2.putText(frame,'4',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                val = 4
                entry += 1
            if count_defects == 4:
                cv2.putText(frame,'5',(100,99), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2,cv2.LINE_AA)
                val = 5
                entry += 1
            if val != -10:
                data.append(val)
        except:
            cv2.putText(frame,"-",(200,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        
        cv2.imshow('frame',frame)

        cv2.imshow('comb', mask)

        cv2.imshow('game board', img)
        # cv2.imshow('res',res)
        i = 0
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        
        #-------------------------------------------------------------------------------------------

        # finding the most occured number from last 100 loop cycles
        if entry == 100:
            num = np.bincount(data).argmax()
            flag = 1
            data = []
            entry = 0

        #-------------------------------------------------------------------------------------------

        if turn%2 == 1 and flag == 1:
            move = num
            flag = 0
            
            #To ensure value input in in range 1-7 
            if move in range(1,10): 
                if valid_loc(board, move):
                    user_move(board, move, opp1)
                    img = mark(img, move, opp1)
                    spawn_board(board)
                    turn = turn+1
            
                else:
                    print("Invalid! The position is already filled\nEnter value again")

                #The player won't win anyway xd
                '''if endgame(board, opp1):
                    print("!!!PLAYER WINS!!!")
                    game_over = True
                else:'''
            
            else:
                print("Invalid input!!!\nPlease Enter value in range 1-9")

        elif turn%2 == 0:
            if turn==0: # if computer plays first, the first move will be random 
                        # to reduce time complexity and increase the chances of tie for player
                com_move = random.randrange(1,10)
                board[com_move] = p1
                spawn_board(board)
                turn = turn+1

            else:
                com_move = best_move(board, p1, opp1)
                board[com_move] = p1
                print("The computer played the move " + str(com_move) + "\n")
                spawn_board(board)
                img = mark(img, com_move, p1)

                if endgame(board, p1):
                    print("!!!YOU LOST hehehe!!!")
                    result = 1
                    game_over = True
                else:
                    turn = turn + 1

print("!!!<THANKS FOR PLAYING>!!!")

cv2.destroyAllWindows()

img2 = np.zeros((512,512,3), np.uint8)
if result == 0:
    cv2.putText(img2,'Match Tie',(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(img2,'Thanks for playing ;)',(50,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
elif result == 1:
    cv2.putText(img2,'Computer Won',(30,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img2,'Thanks for playing ;)',(50,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
cv2.imshow('game board', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
