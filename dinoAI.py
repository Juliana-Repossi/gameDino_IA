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
RENDER_GAME = False

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
    score,weights = x
    return score

class alg_genetic:
    def __init__(self,size, max_iter, qtd_generation, percent_selection,percent_cross, percent_mutation):
        self.size = size
        self.max_iter = max_iter
        self.qtd_generation = qtd_generation
        self.time_max = 43200 #12 horas
        
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

        #manter o tamanho da amostra
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

        global aiPlayer

        #inicializa um vetor inicial com valores aleatórios e valores bons ja encontrados
        list_weights = np.random.randint(-1000, 1000, (self.qtd_generation,self.size))
        
        row = np.array([  531 ,  385 ,  433 ,   69 ,   24  , -47  , 188 ,  722,   668 ,  306,  -787 ,-1121,  -122  ,  56  ,-448  , 191  , 778   , -8  ,-789  ,-542])
        list_weights = np.r_[list_weights,[row]]
        
        
        #verificar a convergencia
        conv = self.convergent(list_weights)
        
        while not conv and iter < self.max_iter and end-start <= self.time_max:
    
            results = []
                        
            for weights in list_weights:
                
                #concatenar resultado e pesos
                aiPlayer = KeySimplestClassifier(weights)
                res, value = manyPlaysResults(3)
                results += [(value,weights)]
                
            #porcentagem da população com os melhores valores 
            elite, best_score, best_weights = self.elitism(results)
            
            #adicionar o melhor da geração ao grafico
            self.graphic += [best_score]

            print(best_score)

            #guarda os melhores valores
            if (best_score > self.best_score):
                print('\n new best_score\n')
                print(best_score)
                print('\n new best_weights\n')
                print(best_weights)
                self.best_score = best_score
                self.best_weights = best_weights


            # faz a seleção
            selections = self.selection(results, len(list_weights) - len(elite))
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

            #aplicar transformação de valor nominal para ordinal no obType            
            if isinstance(obType, Bird) and obHeight > 40:
                userInput = aiPlayer.keySelector(distance, obHeight, game_speed, -700)

            elif isinstance(obType, Bird) and obHeight <= 40:
                userInput = aiPlayer.keySelector(distance, obHeight, game_speed, 0)

            elif isinstance(obType, SmallCactus):
                userInput = aiPlayer.keySelector(distance, obHeight, game_speed, 0)

            elif isinstance(obType, LargeCactus):
                userInput = aiPlayer.keySelector(distance, obHeight, game_speed, 0)

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



# roda o jogo varias vezes para medir o resultado
def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())

def main():

    #inicializando a heurística (size,max_iter,qtd_gerac,selecao,crossfit,mutação)
    meta_alg_genetic = alg_genetic(N_NEURONIOS**2 + N_NEURONIOS, 200, 1, 0.2, 0.9, 0.2)
    
    #fase de aprendizagem
    meta_alg_genetic.metaheuristica_genetic()
    


    print("\n\n ---------- Resultado final ----------- \n\n")
    print(" \n ---------- Melhor Pontuação ----------- \n")
    print(meta_alg_genetic.best_score)
    print("\n ---------- Melhor Array de Pesos ----------- \n")
    print(meta_alg_genetic.best_weights)

    global aiPlayer
    aiPlayer = KeySimplestClassifier(meta_alg_genetic.best_weights)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print("\n ---------- Resultados ----------- \n")
    print(res, npRes.mean(), npRes.std(), value)

main()