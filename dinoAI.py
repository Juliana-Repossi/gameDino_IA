import pygame
import os
import random
import time
import numpy as np
from sys import exit
from scipy import stats
import math

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
#GAME_MODE = "HUMAN_MODE"
RENDER_GAME = True

# Global Constants
#NUMERO DE NEURÔNIOS - CARACTERISTICAS
N_NEURONIOS = 4
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
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
    return x[0]

class alg_genetic:
    def __init__(self,qtd_weights, max_iter, qtd_generation, percent_selection,percent_cross, percent_mutation):
        self.qtd_weights = qtd_weights
        self.best_score = 0
        self.best_weights = []
        self.time_max = 43200 #12 horas
        self.max_iter = max_iter
        self.qtd_generation = qtd_generation
        self.percent_selection = percent_selection
        self.tam = self.qtd_weights**2 + self.qtd_weights
        self.percent_cross = percent_cross
        self.percent_mutation = percent_mutation

    def convergent(self,list_weights):
        conv = False
        if list_weights != []:
            base = list_weights[0]
            i = 1
            while i < self.qtd_weights:
                if (base != list_weights[i]).any:
                    return False
                i += 1
            return True

    def elitism (self, results):
        n = math.floor((self.percent_selection/100)*len(self.qtd_generation))
        if n < 1:
            n = 1
        bests = sorted (results, key = first, reverse = True)[:n]
        best_score,best_weights = best[0]
        elite = [v for s,v in best]
        return elite,best_score,best_weights

    def crossover(dad,mom):
        r = random.randint(0, len(dad) - 1)
        son = dad[:r]+mom[r:]
        daug = mom[:r]+dad[r:]
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

            ##### ???????????
            if rand <= self.taxa_cross:
                parent1, parent2 = crossover(parent1, parent2)            
    
            new_weights = new_weights + [parent1, parent2]
            
        return new_weights

    def mutation (weights):
        copy_weights = weights.copy()
        rand = random.randint(0, len(copy_weights) - 1)
        
        if copy_weights[rand] > 0:
            r = random.uniform(0,1)
            if r > 0.5:
                copy_weights[rand] = copy_weights[rand] + 1
            else:
                copy_weights[rand] = copy_weights[rand] - 1
        else:
            copy_weights[rand] = copy_weights[rand] + 1
            
        return copy_weights

    def mutation_weights (self, list_weights):
        ind = 0
        for weights in list_weights:
            rand = random.uniform(0, 1)

            if rand <= self.percent_mutation:
                mutated = mutation(weights)
                if state_size(mutated, items) <= max_size:
                    list_weights[ind] = mutated
                    
            ind+=1
            
        return list_weights   

    def metaheuristica_genetic(self):

        #valores de iteração
        start = time.process_time()
        iter = 0    
        end = 0

        global aiPlayer

        #inicializa um vetor inicial de pesos aleatoriamente // qtd de pesos é n³
        list_weights = np.random.randint(0, 1000, (self.qtd_generation,self.tam))

        print(list_weights)
        #verificar a convergencia
        conv = self.convergent(list_weights)
        
        while not conv and iter < self.max_iter and end-start <= self.time_max:

            results = []
            
            for weights in list_weights:
                
                #testar combinações geradas --> jogar --> array de scores (score , pesos)
                aiPlayer = KeySimplestClassifier(weights)
                res, score = manyPlaysResults(3)
                results += (score,weights)

            #porcentagem da população com os melhores valores 
            elite, best_score, best_weights = elitism(results)

            #guarda os melhores valores
            if (best_score > self.best_score):
                self.best_score = best_score
                self.best_weights = best_weights

            #gerando automaticamente os valores faltantes para compor a prox exec !!!!!!!FEZ COM ROLETA ????
            selection = np.random.randint(0, 1000, (self.qtd_generation - len(elite),self.tam))
            # faz o cross over - duvida na taxa
            crossed = crossover(selected)
            #aplica mutação
            mutated = mutation_step(crossed)
            list_weights = elite + mutated
            conv = self.convergent(list_weights)
            iter+=1
            end = time.process_time()
        
        return best_score, best_weights


# exemplo de implementação
# distance - Distância do dino até o próximo obstáculo
# obHeight - Altura do próximo obstáculo
# speed - Velocidade atual do jogo
# obType - Tipo de obstáculo que pode ser SmallCactus, LargeCactus ou Bird 
#           tendo este último três variações de altura baixo,médio e alto.

class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.neuronios = N_NEURONIOS

    def keySelector(self, distance, obHeight, speed, obType):

        valor_neuronios = []
        result = 0
        
        #implementação de redes neurais
        for i in range(N_NEURONIOS):
            valor_neuronios[i] = distance*self.state[i*N_NEURONIOS+0] + obHeight*self.state[i*N_NEURONIOS+1] + speed*self.state[i*N_NEURONIOS+2] + obType*self.state[i*N_NEURONIOS+3]          

        #para cada neurônio
        for i in range(N_NEURONIOS):
            result = result + valor_neuronios[i]*self.state[N_NEURONIOS*N_NEURONIOS + i]
       
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


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font('freesansbold.ttf', 20)

    player = Dinosaur()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

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

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

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

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)

        if RENDER_GAME:
            player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)


        if RENDER_GAME:
            background()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                if RENDER_GAME:
                    pygame.time.delay(2000)
                death_count += 1
                return points


# Change State Operator
def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1:]


# Neighborhood
def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        ds = random.randint(1, 10)
        dd = random.randint(1, 100)
        new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                      change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent
def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(3)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value


# roda o jogo varias vezes para medir o resultado
def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())

# Na função main (linha 429 a 439) é possível observar que existe uma implementação da
# subida de gradiente para alterar o vetor de estados. Depois são feitas 30 rodadas do jogo para
# avaliar o processo de otimização. O estado é o que sua metaheuristica deve otimizar para
# jogar, nesse caso foi utilizado um vetor de tuplas como um estado, porém em outras
# implementações como, por exemplo, utilizando redes neurais o estado será o vetor de pesos
# dos nós da rede.p
def main():

    #heurística buscando o melhor resultado
    metah_genetic = alg_genetic(N_NEURONIOS, 5000, 5, 0.5,0.6, 0.2)
    best_score, best_weights = metah_genetic.metaheuristica_genetic()

main()