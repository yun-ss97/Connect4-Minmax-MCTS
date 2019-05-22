# 참조한 MinMax 공개코드 URL
# https://www.youtube.com/watch?v=MMLtza3CZFM&list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV&index=6
# https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
# 참조한 MCTS 공개코드 URL
# https://repl.it/talk/challenge/Connect-4-AI-using-Monte-Carlo-Tree-Search/10640


import numpy as np
import random
import math
import time
import copy


#이윤승
#메인 실행 코드 구현 기여

ROW_COUNT = 6
COLUMN_COUNT = 7

EMPTY = 0
PLAYER = 0
AI = 1

PLAYER_PIECE = 1  # human turn 일 때는 '1'로 표현
AI_PIECE = 2  # AI turn 일 때는 '2'로 표현

WINDOW_LENGTH = 4  # 4개가 차면 이기는걸로


def create_board():  # 6x7 의 게임판을 만듦.
    board = np.zeros((6, 7))
    return board


def drop_piece(board, row, col, piece):  # 행과 열에 해당하는 특정 지점에 piece 를 착수
    board[row][col] = piece


def is_valid_location(board, col, round):  # 해당 column 이 다 차있는 것이 아니면 수를 놓을 수 있으므로 valid 한 위치인지 확인
    if round == 0:  # 첫 라운드에 column 4에 못 놓음
        board[5][3] = 3
    return board[5][col] == 0


def get_next_open_row(board, col):  # 착수 가능한 row(아직 비어있는 row)
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):  # 직관적으로 보기 쉽도록 np.array 를 flip 함.
    print(np.flip(board, 0))


def winning_move(board, piece):  # 이기는 경우의 수(가로, 세로, 대각선(+/-) 상황을 모두 포함)
    # check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # check for vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):  # 꼭대기에서 시작할 수 없으니까 -3 해주어야 함
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # check positively sloped diagnols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    # check negatively sloped diagnols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

#이윤승
#MinMax 알고리즘 구현 기여

#Minmax 알고리즘 공개 코드 참고 URL: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
#공개코드의 maximizingPlayer 와 MinimizingPlayer 가 작동하는 방식, Alpha-Beta Pruing 기법 참고
#착수 가능한 column 의 탐색 순서, 휴리스틱 함수를 차별

class Minmax:

    #정채연
    #heuristic function 구현 기여
    def heuristic(board, piece, opp_piece):  # 각 전략에 따른 점수를 계산함

        score = score_opp = final_score = 0
        # xx_count은 ii 가 range(4)만큼 for문을 돌면서 자신의 돌과 같은 수를 찾을 때마다 값이 증가하는데,
        # xx_count에 (10-i)가 가중치로 곱해져 연속된 돌을 찾을수록, 즉, 같은 돌이 가까이에 있을 수록 더 큰 점수를 부여하게 된다.
        e_count = w_count = s_count = ne_count = nw_count = se_count = sw_count = 0

        valid_locations = Minmax.get_valid_locations(board)

        for c in valid_locations:

            r = get_next_open_row(board, c)
            b_copy = board.copy()
            drop_piece(b_copy, r, c, AI_PIECE)

            # 가운데 column 일수록 더 높은 점수 주기
            if c == 3:
                score += 600
            if c == 2 or c == 4:
                score += 300
            if c == 1 or c == 5:
                pass

            # 내꺼 먼저
            # horizontally 3 in a row 이고 그 양옆이 비어있다면 점수 더 주기
            if r == 0:
                for cc in range(1, 4):
                    if b_copy[r][cc] == piece and b_copy[r][cc + 1] == piece and b_copy[r][cc + 2] == piece and \
                            b_copy[r][cc + 3] == 0 and b_copy[r][cc - 1] == 0:
                        score += 1000

            for i in range(1, 4):  # for every possible winning line

                # 동
                if 0 <= c <= 3 and b_copy[r][c + i] == piece:
                    e_count += 1

                    score += 5 * (10 - i) + e_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if e_count == 3:
                        score += 2000

                # 서
                if 3 <= c <= 6 and b_copy[r][c - i] == piece:
                    w_count += 1
                    score += 5 * (10 - i) + w_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if w_count == 3:
                        score += 2000

                # 남
                if 3 <= r <= 5 and b_copy[r - i][c] == piece:
                    s_count += 1
                    score += 5 * (10 - i) + s_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if s_count == 3:
                        score += 2000

                # 북 필요없음

                # 북동
                if 0 <= r <= 2 and 0 <= c <= 3 and b_copy[r + i][c + i] == piece:
                    ne_count += 1
                    score += 5 * (10 - i) + ne_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if ne_count == 3:
                        score += 2000

                # 북서
                if 0 <= r <= 2 and 3 <= c <= 6 and b_copy[r + i][c - i] == piece:
                    nw_count += 1
                    score += (5 * (10 - i) + nw_count * (10 - i)) * 2
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if nw_count == 3:
                        score += 2000

                # 남동
                if 3 <= r <= 5 and 0 <= c <= 3 and b_copy[r - i][c + i] == piece:
                    se_count += 1
                    score += 5 * (10 - i) + se_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if se_count == 3:
                        score += 2000

                # 남서
                if 3 <= r <= 5 and 3 <= c <= 6 and b_copy[r - i][c - i] == piece:
                    sw_count += 1
                    score += (5 * (10 - i) + sw_count * (10 - i)) * 2
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if sw_count == 3:
                        score += 2000

            # 상대방 견제 // 상대방이 돌을 놨을 때 상대방이 유리해질수록 내가 그 점에 놔야한다

            # copy board 초기화
            b_copy = board.copy()
            drop_piece(b_copy, r, c, PLAYER_PIECE)

            # count 초기화
            e_count = w_count = s_count = ne_count = nw_count = se_count = sw_count = 0

            # 상대 3개 연속 양 옆 비어있을때
            if 1 <= c <= 3 and b_copy[r][c] == opp_piece and b_copy[r][c + 1] == opp_piece and b_copy[r][
                c + 2] == opp_piece and b_copy[r][c + 3] == 0 and b_copy[r][c - 1] == 0:
                score_opp += 1000  # 무조건 여기 놔야하니까

            for i in range(1, 4):  # for every possible winning line
                # 동
                if 0 <= c <= 3 and b_copy[r][c + i] == opp_piece:
                    e_count += 1

                    score_opp += 5 * (10 - i) + e_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if e_count == 3:
                        score_opp += 20000

                # 서
                if 3 <= c <= 6 and b_copy[r][c - i] == opp_piece:
                    w_count += 1
                    score_opp += 5 * (10 - i) + w_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if w_count == 3:
                        score_opp += 20000

                # 남
                if 3 <= r <= 5 and b_copy[r - i][c] == opp_piece:
                    s_count += 1
                    score_opp += 5 * (10 - i) + s_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if s_count == 3:
                        score_opp += 20000

                # 북 필요없음
                # 북동
                if 0 <= r <= 2 and 0 <= c <= 3 and b_copy[r + i][c + i] == opp_piece:
                    ne_count += 1
                    score_opp += 5 * (10 - i) + ne_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if ne_count == 3:
                        score_opp += 20000

                # 북서
                if 0 <= r <= 2 and 3 <= c <= 6 and b_copy[r + i][c - i] == opp_piece:
                    nw_count += 1
                    score_opp += (5 * (10 - i) + nw_count * (10 - i)) * 2
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if nw_count == 3:
                        score_opp += 20000

                # 남동
                if 3 <= r <= 5 and 0 <= c <= 3 and b_copy[r - i][c + i] == opp_piece:
                    se_count += 1
                    score_opp += 5 * (10 - i) + se_count * (10 - i)
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if se_count == 3:
                        score_opp += 20000

                # 남서
                if 3 <= r <= 5 and 3 <= c <= 6 and b_copy[r - i][c - i] == opp_piece:
                    sw_count += 1
                    score_opp += (5 * (10 - i) + sw_count * (10 - i)) * 2
                    # 4개 연속으로 바로 이기면 매우 큰 점수
                    if sw_count == 3:
                        score_opp += 20000

            final_score = score + score_opp
            print("column %d의 heuristic function score: %d" % (c + 1, final_score))

        return final_score

    def get_valid_locations(board):  # 착수할 수 있는 지점들을 모아놓음.
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col, round):
                valid_locations.append(col)
        return valid_locations

    def is_terminal_node(board):  # 가장 마지막 노드인지 확인하는 함수
        return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(
            Minmax.get_valid_locations(board)) == 0

    # 위키피디아 수도코드 참고 , minmax + alpha beta pruning 추가함.
    def minimax(board, depth, alpha, beta, maximizingPlayer):

        valid_locations = Minmax.get_valid_locations(board)
        is_terminal = Minmax.is_terminal_node(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(board, AI_PIECE):  # AI가 이기는 경우
                    return (None, 10000000000000)  # 밑에 return 형식과 맞춤.
                elif winning_move(board, PLAYER_PIECE):  # Player가 이기는 경우
                    return (None, -1000000000000)
                else:  # game is over, no more valid moves
                    return (None, 0)
            else:  # depth가 0일 때는 heuristic을 고려함
                return (None, Minmax.heuristic(board, AI_PIECE, PLAYER_PIECE))

        if maximizingPlayer:
            value = -math.inf
            valid_locations = random.sample(valid_locations, len(valid_locations))

            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                drop_piece(b_copy, row, col, AI_PIECE)

                new_score = Minmax.minimax(b_copy, depth - 1, alpha, beta, False)[1]

                if new_score > value:
                    value = new_score
                    column = col

                # alpha cut off: 불필요한 연산을 잘라냄.
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return column, value



        else:  # minimizing player
            value = math.inf
            valid_locations = random.sample(valid_locations, len(valid_locations))

            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                drop_piece(b_copy, row, col, PLAYER_PIECE)

                new_score = Minmax.minimax(b_copy, depth - 1, alpha, beta, True)[1]

                if new_score < value:
                    value = new_score
                    column = col
                # beta cut off: 불필요한 연산을 잘라냄.
                beta = min(beta, value)
                if alpha >= beta:
                    break

            return column, value


#김수빈
#MCTS+UCT function 구현 기여
# the following code was adapted and modified from: http://mcts.ai/code/python.html
# MCTS 공개 코드 참고 URL : # https://repl.it/talk/challenge/Connect-4-AI-using-Monte-Carlo-Tree-Search/10640
# 탐색 알고리즘 작동 원리 참고
# MCTS 내 selection 에 사용되는 heuristic인 UCT function 차별화
class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.state = state.Clone()
        self.parent = parent
        self.move = move
        self.untriedMoves = state.getMoves()
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.player = state.player

    def selection(self):
        # return child with largest UCT value
        # selection function으로 계산된 자식노드들의 점수 중, 가장 큰 값 반환
        # 논문 참고, improved 된 UCT formula 이용
        # < Analysing And Improving The Knowledge-based Fast Evolutionary MCTS Algorithm>
        # The factor 1/4 is an upper bound on the variance of a Bernoulli random variable.
        # It should be noted that Auer et al. were not able to prove a regret bound for
        # UCB1-Tuned, but experimentally found that it performed better.

        """
        공개코드 원본 UCT(UCB1) heuristic function
        s = lambda x: x.wins / x.visits + np.sqrt(2 * np.log(self.visits) / x.visits)
        return sorted(self.childNodes, key=s)[-1]
        """

        # 우리의 heuristic function - improved UCT - 공개코드와의 차이점
        v_result = list(map(lambda x: x.wins, self.childNodes))
        var_of_action = list(
            map(lambda x: (sum(v_result) ** 2) / x.visits - (sum(v_result) / x.visits) ** 2, self.childNodes))
        UCT = list(map(lambda x: np.sqrt(2 * np.log(self.visits) / x.visits), self.childNodes))
        UCT_improved_term = []
        for i in range(len(UCT)):
            res = UCT[i] + var_of_action[0]
            UCT_improved_term.append(res)
        s = lambda x: x.wins / x.visits + 1 * np.sqrt(np.log(self.visits) / x.visits) * np.sqrt(
            min([1 / 4], UCT_improved_term)[0])
        return sorted(self.childNodes, key=s)[-1]


    def expand(self, move, state):
        # return child when move is taken
        # remove move from current node
        child = Node(move=move, parent=self, state=state)
        self.untriedMoves.remove(move)
        self.childNodes.append(child)
        return child

    def update(self, result):
        self.wins += result
        self.visits += 1


def MCTS(currentState, itermax, currentNode=None, timeout=120):
    # currentNode가 None 인 경우 수행 되는 부분
    rootnode = Node(state=currentState)
    # None이 아니면 현 노드를 rootnode로 지정
    if currentNode is not None: rootnode = currentNode

    start = time.process_time()

    for i in range(itermax):
        node = rootnode
        state = currentState.Clone()

        while node.untriedMoves == [] and node.childNodes != []:
            # selection
            node = node.selection()
            state.move(node.move)
        # node.selection()의 결과는 selection에 의해 선택된 자식 노드
        # node.move는 돌을 놓을 열 번호가 되고, state.move에서 인자로 사용되어 그 열 정보(해당 열에서 돌을 놓을 수 있는 행 번호 / 열에 놓인 돌 개수 등) 업데이트)

        # expand
        #One (or more) child nodes are selected and added to the tree, according to the reachable states from the current node.
        # exploration / random 열 번호 m에서 시작
        # state.move를 통해 state 업데이트
        # node.expand를 통해 untriedMoves에서 탐색된 m 원소 지우고, childNodes에
        # 탐색된 노드 추가
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.move(m)
            node = node.expand(m, state)

        # rollout
        # getMoves함수를 통해 현 상태에서 이동할 수 있는 열 리스트가 출력
        # 그 중 하나를 랜덤으로 정해서 move 실행.
        while state.getMoves():
            state.move(random.choice(state.getMoves()))

        # backpropagate
        while node is not None:
            node.update(state.result(node.player))
            node = node.parent

        # timeout 시간 경과하면 중단
        duration = time.process_time() - start
        if duration > timeout: break

    # childNodes 에서 승률이 높은 순으로 정렬 [::-1]하여 sortedChildNodes로 저장
    s = lambda x: x.wins / x.visits
    sortedChildNodes = sorted(rootnode.childNodes, key=s)[::-1]
    print("AI\'s computed winning percentages")
    for node in sortedChildNodes:
        print('Move: %s    Win Rate: %.2f%%' % (node.move + 1, 100 * node.wins / node.visits))
        # node.move + 1인 이유는 인덱스가 0부터 시작하기 때문
    print('Simulations performed: %s\n' % i)
    return rootnode, sortedChildNodes[0].move
    # 루트 노트와 승률이 높은 childNodes move값 반환


class MCTSboard:
    def __init__(self, ROW, COLUMN, LINE):
        self.bitboard = [0, 0]  # bitboard for each player
        self.dirs = [1, (ROW + 1), (ROW + 1) - 1, (ROW + 1) + 1]  # this is used for bitwise operations
        self.heights = [(ROW + 1) * i for i in range(COLUMN)]  # top empty row for each column
        self.lowest_row = [0] * COLUMN  # number of stones in each row
        self.board = np.zeros((ROW, COLUMN), dtype=int)  # matrix representation of the board (just for printing)
        self.top_row = [(x * (ROW + 1)) - 1 for x in
                        range(1, COLUMN + 1)]  # top row of the board (this will never change)
        self.ROW = ROW
        self.COLUMN = COLUMN
        self.LINE = LINE
        self.player = 1

    def Clone(self):
        clone = MCTSboard(self.ROW, self.COLUMN, self.LINE)
        clone.bitboard = copy.deepcopy(self.bitboard)
        clone.heights = copy.deepcopy(self.heights)
        clone.lowest_row = copy.deepcopy(self.lowest_row)
        clone.board = copy.deepcopy(self.board)
        clone.top_row = copy.deepcopy(self.top_row)
        clone.player = self.player
        return clone

    # 어떤 column에 돌을 놓았을 경우, 바뀌는 설정값
    def move(self, col):
        m2 = 1 << self.heights[col]  # position entry on bitboard
        # <<는 왼쪽 쉬프트 비트 연산. 2를 곱하는 효과
        self.heights[col] += 1  # update top empty row for column
        self.player ^= 1  # XOR 연산
        self.bitboard[self.player] ^= m2  # XOR operation to insert stone in player's bitboard
        self.board[self.lowest_row[col]][col] = self.player + 1  # update entry in matrix (only for printing)
        self.lowest_row[col] += 1  # update number of stones in column

    def result(self, player):
        if self.winner(player):
            return 1  # player wins
        elif self.winner(player ^ 1):
            return 0  # if opponent wins
        elif self.draw():
            return 0.5
        # checks if column is full

    # evaluate board, find out if there's a winner
    def winner(self, color):
        for d in self.dirs:
            bb = self.bitboard[color]
            for i in range(1, self.LINE):
                bb &= self.bitboard[color] >> (i * d)
            if (bb != 0): return True
        return False

    def draw(self):  # is it draw?
        return not self.getMoves() and not self.winner(self.player) and not self.winner(self.player ^ 1)

    # returns list of available moves
    def getMoves(self):
        if self.winner(self.player) or self.winner(
                self.player ^ 1): return []  # if terminal state( 이긴 상태이므로 종료), return empty list

        listMoves = []
        for i in range(self.COLUMN):
            if self.lowest_row[i] < self.ROW:
                listMoves.append(i)
        return listMoves


def goto_childNode(node, board, move):
    for childnode in node.childNodes:
        if childnode.move == move:
            return childnode
    return Node(state=board)


# 게임 판을 만듦
board = create_board()
print("<<Initial Game Board>>")
print_board(board)
print("-------------------------")
game_over = False  # game 이 종료되면 True 로 바꾸어줌.

seq = input("Do you want to go first or not? (Y/N)")

while seq != 'y' or seq != 'Y' or seq != 'n' or seq != 'N':
    if seq == 'y' or seq == 'Y':
        print("You are first player")
        turn = PLAYER
        break

    elif seq == 'n' or seq == 'N':
        print("You are second player")
        turn = AI
        break

    else:
        print('잘못 입력했습니다.')
        seq = input("Do you want to go first or not? (Y/N)")

oROW, oCOLUMN = 6, 7  # change size of board here
oLINE = 4  # change number of in-a-row here
c4 = MCTSboard(oROW, oCOLUMN, oLINE)  # create MCTSboard object
node = Node(state=c4)

itermax = 10000
timeout = 120
round = 0
is_draw = 0


def alg():  # 알고리즘을 랜덤으로 골라주는 함수
    # alg = random.randint(1, 2)
    # return alg
    alg = random.randint(1,10)
    if 1 <= alg <= 8: # MCTS 8
        return 1
    else:
        return 2 # Minmax 2
    #return 1 #MCTS만 하도록, 2로 변경시 MinMax
"""
def alg(): # 알고리즘을 랜덤으로 골라주는 함수
    alg = random.randint(1,2)
    return 1
"""

while not game_over:  # turn 의 나머지가 0이면 PLAYER 차례이고, 나머지가 1이면 AI 차례임.
    is_draw = 0

    # if turn == PLAYER:
    if turn % 2 == 0:
        col = int(input("Player make your selection (1-7):")) - 1
        while col == 3 and turn == 0:
            print('선택이 불가합니다.')
            col = int(input("Player make your selection (1-7):")) - 1

        while col + 1 < 1 or col + 1 > 7:
            print('선택이 불가합니다.')
            col = int(input("Player make your selection (1-7):")) - 1

        if is_valid_location(board, col, round):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_PIECE)
            c4.move(col)
        node = goto_childNode(node, c4, col)

        if round == 0:  # column 4 막아놨던 것 원상복구
            board[5][3] = 0

        print_board(board)
        print("-------------------------")

        for c in range(7):
            if board[5][c] != 0:
                is_draw += 1

        if winning_move(board, PLAYER_PIECE):
            print("player wins")
            game_over = True

        elif is_draw == 7:
            print("DRAW")
            game_over = True

        turn += 1
        round += 1

    # Ask for player 2 input
    elif turn % 2 == 1 and not game_over:

        if alg() == 1:
            print("선택된 방법은 MCTS 입니다.")
            # MCTS
            if round == 0:
                for i in range(6):
                    c4.move(3)
                    node = goto_childNode(node, c4, 3)

            print('AI\'S thinking...')
            start_time = time.time()
            node, col = MCTS(c4, itermax, currentNode=node, timeout=timeout)

            print('AI played column %s\n' % (col + 1))
            print("---- %.2f seconds ----" % (time.time() - start_time))  # 시간 체크)

            if round == 0:
                c4 = MCTSboard(oROW, oCOLUMN, oLINE)  # create MCTSboard object
                node = Node(state=c4)
            #     round += 1

            if is_valid_location(board, col, round):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col,AI_PIECE)
                c4.move(col)
            node = goto_childNode(node, c4, col)

            if round == 0:  # column 4 막아놨던 것 원상복구
                board[5][3] = 0

            print_board(board)
            print("-------------------------")


            for c in range(7):
                if board[5][c] != 0:
                    is_draw += 1

            if winning_move(board, AI_PIECE):
                print("AI wins")
                game_over = True

            elif is_draw == 7:
                print("DRAW")
                game_over = True

            # node = goto_childNode(node, c4, col)
            turn += 1
            round += 1


        else:

            print("선택된 방법은 Minmax + alpha-beta pruning 입니다.")
            # Minmax Algorithm

            start_time = time.time()
            col, minimax_score = Minmax.minimax(board, 6, -math.inf, math.inf, True)
            print("---- %.2f seconds ----" % (time.time() - start_time))  # 시간 체크)

            if is_valid_location(board, col, round):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)
                c4.move(col)
            node = goto_childNode(node, c4, col)

            if round == 0:  # column 4 막아놨던 것 원상복구
                board[5][3] = 0

            print_board(board)
            print("[AI CHOICE] minmax score: %d 이므로 column %d 를 선택함" % (minimax_score, col + 1))

            for c in range(7):
                if board[5][c] != 0:
                    is_draw += 1

            if winning_move(board, AI_PIECE):
                print("AI wins")
                game_over = True
            elif is_draw == 7:
                print("DRAW")
                game_over = True

            turn += 1
            round += 1