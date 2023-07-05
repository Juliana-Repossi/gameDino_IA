import pygame
import os
import random
import time
from sys import exit
import math

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
RENDER_GAME = False

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
N_NEURONIOS = 4
if RENDER_GAME:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        pygame.draw.rect(SCREEN, self.color,
                         (self.dino_rect.x, self.dino_rect.y, self.dino_rect.width, self.dino_rect.height), 2)


    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass

def first(x):
    score,weights = x
    return score

class alg_genetic:
    def __init__(self,size, max_iter, qtd_generation, percent_selection,percent_cross, percent_mutation):
        self.size = size
        self.max_iter = max_iter
        self.qtd_generation = qtd_generation
        # self.time_max = 43200 #12 horas
        self.time_max = 33000 # 9 h
           
        self.percent_selection = percent_selection
        self.percent_cross = percent_cross
        self.percent_mutation = percent_mutation

        self.best_score = 0
        self.best_weights = []
        self.graphic = []

    def convergent(self,list_weights):
        conv = False
        if len(list_weights) != 0:
            base = list_weights[0]
            i = 1
            while i < len(list_weights):
                if (base != list_weights[i]).any:
                    return False
                i += 1
            return True

    def elitism (self, results):
        n = math.floor((self.percent_selection)*len(results))
        if n < 1:
            n = 1
        bests = sorted (results,key=first, reverse = True)[:n]
        best_score,best_weights = bests[0]
        elite = [v for s,v in bests]
        return elite,best_score,best_weights

    def scores_total_value(self,results_weights):
        total_scores = 0
        for score,weights in results_weights:
            if score < 0:
                score = 0
            total_scores = total_scores + score
        return total_scores

    def roulette_construction(self,results_weights):
        aux_states = []
        roulette = []
        total_value = self.scores_total_value(results_weights)

        for score,weights in results_weights:
            if score <= 0:
                # scores não positivos não são a realidade dessa aplicação 
                continue
            
            value = score
            ratio = value/total_value
            
            aux_states.append((ratio,weights))
    
        acc_value = 0
        for score,weights in aux_states:
            acc_value = acc_value + score
            s = (acc_value,weights)
            roulette.append(s)
        return roulette
    
    def roulette_run (self,rounds, roulette):
        if roulette == []:
            return []
        selected = []
        while len(selected) < rounds:
            r = random.uniform(0,1)
            for space,weights in roulette:
                if r <= space:
                    selected.append(weights)
                    break
        return selected
    
    def selection(self,results_weights,n):
        aux_population = self.roulette_construction(results_weights)
        new_population = self.roulette_run(n, aux_population)
        return new_population

    def crossover(self,dad,mom):
        r = random.randint(2, len(dad) - 1)
        son = np.concatenate([dad[:r], mom[r:]])
        daug = np.concatenate([mom[:r], dad[r:]])
        return son, daug

    def crossover_weights (self, list_weights):
        new_weights = []
        
        for _ in range (round(len(list_weights)/2)):
            rand = random.uniform(0, 1)
            #sorteando um lista de pesos para sofrer cross
            fst_ind = random.randint(0, len(list_weights) - 1)
            scd_ind = random.randint(0, len(list_weights) - 1)

            parent1 = list_weights[fst_ind] 
            parent2 = list_weights[scd_ind]

            if fst_ind != scd_ind and rand <= self.percent_cross:

                parent1, parent2 = self.crossover(parent1, parent2)            
        
            new_weights = new_weights + [parent1, parent2]

        #manter o tamanho da amostra - escolha de projeto
        for i in range(len(list_weights)-len(new_weights)):
          ind = random.randint(0, len(list_weights) - 1)
          new_weights = new_weights + [list_weights[ind]]
            
        return new_weights

    def mutation (self, weights):
        copy_weights = weights.copy()
        rand = random.randint(0, len(copy_weights) - 1)
        
        r = random.uniform(0,1)
        if r > 0.5:
            copy_weights[rand] = copy_weights[rand] + 1
        else:
            copy_weights[rand] = copy_weights[rand] - 1
                    
        return copy_weights

    def mutation_weights (self, list_weights):
        ind = 0
        for weights in list_weights:
            rand = random.uniform(0, 1)

            if rand <= self.percent_mutation:
                mutated = self.mutation(weights)
                list_weights[ind] = mutated
                    
            ind+=1
            
        return list_weights   

    def metaheuristica_genetic(self):

        #valores de iteração
        start = time.process_time()
        iter = 0    
        end = 0        

        #inicializa um vetor inicial com valores aleatórios e valores bons ja encontrados
        list_weights = np.random.randint(-1000, 1000, (self.qtd_generation,self.size))
        

        bons =  np.array([
        [ 511 , 360 , 481 ,  79 ,-174,  102,  243,  942,  668,  296, -647, -713 ,-97 ,  71, -403,  151,  788,  -48 ,-769 ,-615],
        [ 521,  350 , 481 ,-101,  -34,   62,  233,  932,  678,  336, -657, -603, -107,  -19, -463,   51,  788 ,   2 ,-779 ,-632],
        [ 511,  363,  481, -118,  -24,  202,  193,  959,  668,  329, -597, -663,  -97,  18, -350,  141,  778,  -48 ,-762, -638],
        [ 508,  366,  428,  179, -174,   66,  196,  692,  668,  296 ,-797, -813,  -97,   74,  -453,  148,  788,  -54 ,-769, -622],
        [ 511,  360,  484, -115,  -24,   92,  220,  959,  668,  329, -597, -663,  -97,   18, -350,  141,  778,  -48, -762, -638],
        [ 508,  366,  428,  179, -174,   66,  196,  692,  668,  296, -797, -813,  -97,   74, -453,  148,  788,  -54, -769, -622],
        [ 511,  360,  481, -121,  -24,  202,  193,  892,  668,  346, -597, -613,  -97,   21, -553,  151,  788,   52, -769 ,-622],
        [ 511,  370,  401,   -1, -194,  162,  223,  962,  668,  326, -637, -663,  -97, 1, -433,  181, 798,  -58, -779, -622],
        [ 511,  360 , 431,   29 ,-124,   52 , 193 , 892,  668 , 296, -597 ,-663 , -97,   21 , -403 , 151 , 788,  -48 ,-769 ,-572],
        [ 511,  360,  481,  79, -174,  102 , 243 , 942 , 668,  296, -647 ,-713 , -97 ,  71 , -403,  151 , 788 , -48, -769 ,-615],
        [ 511,  360,  431 , -21,  -74 , 202,  193 , 892 , 668 , 346 ,-797 ,-763,  -97 ,  21 , -353 ,  51 , 788 ,   2 ,-769 ,-622],
        [ 511 , 360 , 481 ,-121 , -24,  202 , 193 , 892,  668 , 346 ,-597, -613,  -97 ,  21 , -553 , 151 , 788 ,  52 ,-769 ,-622],
        [ 511 , 360,  481 ,-121,  -24,  202 , 193,  892,  668 , 346 ,-597 ,-613 , -97 ,  21 , -553 , 151 , 788 ,  52 ,-769 ,-622],
        [ 511 , 360,  331,  179 , -74 , 202  ,343 , 842 , 668 , 296 ,-797, -813 , -97  , 71 , -453 ,  51,  788 ,   2 ,-769, -622],
        [ 511 , 360,  431 ,-121  , 26 , -48 ,  43 , 692 , 668 , 296 ,-797, -463  ,-97,  -29  ,-453  ,-99 , 788 ,   2 ,-769 ,-572],
        [ 511,  360,  361 ,  29, -174  , 72 , 193 , 942 , 668 , 306, -597, -673, -107  , 21 , -413 , 151 , 778 , -38, -779 ,-622],
        [ 511,  370,  371 ,  29, -164 ,  92 , 183 , 932 , 668 , 296, -597, -663 , -97 ,  31 , -403 , 141 , 778 , -48, -759 ,-632],
        [ 511 , 360 , 381 ,  39, -174 , 122 , 183 , 942 , 668 , 306, -577 ,-673 , -97  , 31  ,-423 , 141 , 778 , -68, -759 ,-622],
        [ 511,  360 , 421 ,  69, -184 ,  82 , 233 , 942,  668 , 306 ,-647 ,-653 ,-107  , 41 , -423 , 181 , 788 , -28, -769 ,-582],
        [ 511 , 370,  401 ,  -1, -194 , 162 , 223 , 962 , 668 , 326 ,-637 ,-663 , -97  ,  1  ,-433  ,181 , 798 , -58 ,-779 ,-622],
        [-346   ,30  ,366  ,357 ,-102 , 107 , 733  ,905, -585, -169  ,458  ,778 ,-170 , 234  , -56 , 600  ,782  ,932 ,-401, -549],
        [-345  , 31 , 367 , 357, -103 , 106 , 733 , 906, -586, -169 , 458 , 776 ,-170,  235 ,  -56,  598 , 780 , 931, -400, -549],
        [ 390 ,-880 ,-565  ,396 , 447  ,367,  735, -809 , 452,  381, -892, -744, -461, -604  ,-325, -671, -764,   31 ,-737 ,-865],
        [-347 ,  30  ,367 , 357, -104  ,108 , 733  ,907 ,-584 ,-169,  458  ,776, -171  ,235 ,  -57,  598  ,781  ,930, -399, -548],
        [-346 ,  30 , 367 , 357 ,-103 , 107,  733 , 907, -585, -169 , 458 , 776 ,-171 , 235 ,  -57 , 598 , 781 , 931 ,-400, -549],
        [-346,   30,  367,  357, -103,  107,  733,  907, -585, -169,  458,  776, -171,  235,   -57,  598,  781,  931, -400 ,-549],
        [-346,   30,  367,  357, -103,  107,  733,  907, -585, -169,  458,  776, -171,  235,   -57,  598,  781,  931, -400 ,-549],
        [-346,   30,  367,  357, -103,  107,  733,  907, -585, -169,  458,  776, -171,  235,   -57 , 598 , 781 , 931, -400, -549],
        [-346,   30,  367,  357, -103,  107,  733,  907, -585, -169 , 458, 776, -171, 235,  -57,  598,  781,  931, -400, -549],
        [-449, -400,  682,  905, -947,  496, -682,  606, -321,  844,  739,  999, -296,  232, -901, -135, 641, -240,  642, -665],
        [-449, -400,  682,  900, -942,  496, -692,  606,  -88, -403, -463,  96, -546, -600, -804,  923,  914,   59,  455, -848],
        [ 312,  147,  -83,  647, -672,  965,  536,  -47,  -24,  688,  662,  867,  586,  461,  -656,  908, -587,  -17,  919,  125],
        [ 511,  360,  381,  29, -174,   52,  193,  942,  668,  296, -597, -663, -97,   21, -403,  151,  788,  -48, -769, -622],
        [ 312,  146,  -83,  647, -672,  965,  536,  -47,  -24,  688,  663,  867,  586, 461,  -656,  908, -587,  -17,  919,  125],
        [ 511,  360,  381,  29, -174,   102,  193,  942,  668,  296, -597, -663, -97,   21, -403,  151,  788,  -48, -769, -622],
        [ 390, -880, -570,  396,  437,  377,  730, -809,  452,  381, -892, -739, -466, -604,  -325, -671, -764,   36, -727, -860]
        ])

        list_weights = np.concatenate((bons,list_weights))       
        
        #verificar a convergencia
        conv = self.convergent(list_weights)
    
        while not conv and iter < self.max_iter and end-start <= self.time_max:
    
            i=0
            score = []
            results = []

            score = playGame(list_weights)
         
            for weights in list_weights:
                
                #concatenar resultado e pesos
                results += [(score[i],weights)]
                i +=1

            #print(results)
            
            #porcentagem da população com os melhores valores 
            elite, best_score, best_weights = self.elitism(results)
            #adicionar o melhor da geração ao grafico
            self.graphic += [best_score]

            #guarda os melhores valores
            if (best_score > self.best_score):
                print('\nbest_score\n')
                print(best_score)
                print('\nbest_weights\n')
                print(best_weights)
                self.best_score = best_score
                self.best_weights = best_weights

            # faz a seleção
            selections = self.selection(results, self.qtd_generation - len(elite))
            # faz o cross over 
            crossed = self.crossover_weights(selections)
            #aplica mutação
            mutated = self.mutation_weights(crossed)
            list_weights = elite + mutated
            conv = self.convergent(list_weights)
            iter+=1
            end = time.process_time()
    

class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.neuronios = N_NEURONIOS

    def keySelector(self, distance, obHeight, speed, obType):

        valor_neuronios = []
        result = 0
        
        #implementação de redes neurais
        for i in range(self.neuronios):
            valor_neuronios.append(distance * self.state[i*self.neuronios+0] + obHeight * self.state[i*self.neuronios+1] + speed * self.state[i*self.neuronios+2] + obType * self.state[i*self.neuronios+3])

        i=0

        #para cada neurônio
        for valor in valor_neuronios:
            result = result + valor*self.state[self.neuronios**2 + i]
            i+=1
       
        if result <= 0:
            return "K_DOWN"
        else:
            return "K_UP"

    def updateState(self, state):
        self.state = state


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame(solutions):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font('freesansbold.ttf', 20)

    players = []
    players_classifier = []
    solution_fitness = []
    died = []

    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

    for solution in solutions:
        players.append(Dinosaur())
        players_classifier.append(KeySimplestClassifier(solution))
        solution_fitness.append(0)
        died.append(False)

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if RENDER_GAME:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)


    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def statistics():
        text_1 = font.render(f'Dinosaurs Alive:  {str(died.count(False))}', True, (0, 0, 0))
        text_3 = font.render(f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_3, (50, 480))

    while run and (False in died):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

        for i,player in enumerate(players):
            if not died[i]:
                distance = 1500
                nextObDistance = 2000
                obHeight = 0
                nextObHeight = 0
                obType = 2
                nextObType = 2
                if len(obstacles) != 0:
                    xy = obstacles[0].getXY()
                    distance = xy[0]
                    obHeight = obstacles[0].getHeight()
                    obType = obstacles[0]

                if len(obstacles) == 2:
                    nextxy = obstacles[1].getXY()
                    nextObDistance = nextxy[0]
                    nextObHeight = obstacles[1].getHeight()
                    nextObType = obstacles[1]


                #aplicar transformação de valor nominal para ordinal no obType            
                if isinstance(obType, Bird) and obHeight > 40:
                    userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, -700)

                elif isinstance(obType, Bird) and obHeight <= 40:
                    userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, 0)

                elif isinstance(obType, SmallCactus):
                    userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, 0)

                elif isinstance(obType, LargeCactus):
                    userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, 0)

                else:
                    userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, 0)
               

                player.update(userInput)

                if RENDER_GAME:
                    player.draw(SCREEN)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)


        if RENDER_GAME:
            background()
            statistics()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            for i, player in enumerate(players):
                if player.dino_rect.colliderect(obstacle.rect) and died[i] == False:
                    solution_fitness[i] = points
                    died[i] = True

    return solution_fitness


from scipy import stats
import numpy as np

def manyPlaysResultsTrain(rounds,solutions):
    results = []

    for round in range(rounds):
        results += [playGame(solutions)]

    npResults = np.asarray(results)

    mean_results = np.mean(npResults,axis = 0) - np.std(npResults,axis=0) # axis 0 calcula media da coluna
    return mean_results


def manyPlaysResultsTest(rounds,best_solution):
    results = []
    for round in range(rounds):
        results += [playGame([best_solution])[0]]

    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def main():

    #inicializando a heurística (size,max_iter,qtd_gerac,selecao,crossfit,mutação)
    meta_alg_genetic = alg_genetic(N_NEURONIOS**2 + N_NEURONIOS, 100000, 50000, 0.2, 0.9, 0.9)

    #fase de aprendizado
    meta_alg_genetic.metaheuristica_genetic()
    print(meta_alg_genetic.best_score)
    print(meta_alg_genetic.best_weights)


    print("---------- Resultado final -----------")
    res, value = manyPlaysResultsTest(30, meta_alg_genetic.best_weights)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)

main()
