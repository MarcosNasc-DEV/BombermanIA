import pygame
import sys
import importlib
import random
import hashlib
import json

def gerar_checksum_dados(obj):
    def simplificar(o):
        if isinstance(o, list):
            return [simplificar(x) for x in o]
        elif isinstance(o, dict):
            return {k: simplificar(v) for k, v in o.items()}
        elif hasattr(o, '__dict__'):
            # Para Player e Bomba, selecionamos os atributos essenciais
            if isinstance(o, Player):
                return {
                    'grid_x': o.grid_x,
                    'grid_y': o.grid_y,
                    'max_bombas': o.max_bombas,
                    'bomba_nivel': o.bomba_nivel,
                    'ativo': o.ativo,
                    'time': o.time
                }
            elif isinstance(o, Bomba):
                return {
                    'x': o.x,
                    'y': o.y,
                    'nivel': o.nivel,
                    'explodida': o.explodida,
                    'tempo_explosao': o.tempo_explosao,
                    'tempo_fogo': o.tempo_fogo,
                    'fogo': o.fogo
                }
            else:
                return str(o)
        else:
            return o

    dados_simplificados = simplificar(obj)
    json_data = json.dumps(dados_simplificados, sort_keys=True)
    return hashlib.md5(json_data.encode()).hexdigest()




pygame.init()

# Parâmetros de jogo
TILE_SIZE = 48
ROWS, COLS = 11, 13
HUD_HEIGHT = 60
WIDTH, HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE
TEMPO_MOVIMENTO = 0.1
TEMPO_EXPLOSAO = 4
TEMPO_FOGO = 0.5
MAX_BOMBAS = 5
TEMPO_PARTIDA = 180

PONTOS_BLOCO = 100
PONTOS_POWERUP_COLETADO = 200
PONTOS_POWERUP_DESTRUIDO = -50
PONTOS_MATAR_JOGADOR = 1000
PONTOS_VITORIA = 10000

PROB_BOMBA = 0.12
PROB_FOGO = 0.10

screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
pygame.display.set_caption("Bomberman Arena")

COLOR_BG = (20, 20, 20)
COLOR_GRASS = (80, 150, 80)
COLOR_FLOOR = (160,82,45)
COLOR_BREAKABLE = (100, 100, 100)
COLOR_INDESTRUCTIBLE = (60, 60, 60)
COLOR_BOMB = (0, 0, 0)
COLOR_FIRE = (255, 150, 0)
COLOR_PLAYERS = [(0, 200, 255), (255, 100, 100), (200, 255, 100), (255, 255, 0), (255, 255, 255)]

powerup_bomba_img = pygame.transform.scale(pygame.image.load("pu_bomb.jpg"), (TILE_SIZE, TILE_SIZE))
powerup_fogo_img = pygame.transform.scale(pygame.image.load("pu_fire.jpg"), (TILE_SIZE, TILE_SIZE))
bloco_img = pygame.transform.scale(pygame.image.load("bloco.jpg"), (TILE_SIZE, TILE_SIZE))

# bomba_sprite.jpg com 3 frames de 48x48 lado a lado
bomba_sprite = pygame.image.load("sprite_bomb_no_bg.png")
bomb_frames = [
    pygame.transform.scale(bomba_sprite.subsurface(pygame.Rect(i * 48, 0, 48, 48)), (TILE_SIZE, TILE_SIZE))
    for i in range(3)
]

# explosao.png com 5 frames de 48x48 lado a lado
explosao_sprite = pygame.image.load("sprite_explosao.png").convert_alpha()
explosao_frames = [
    explosao_sprite.subsurface(pygame.Rect(i * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE))
    for i in range(5)
]



mapa = [
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]

font = pygame.font.SysFont("Arial", 16)
font_vitoria = pygame.font.SysFont("Arial", 20, bold=True)

def criar_powerup(x, y, prob_bomba=PROB_BOMBA, prob_fogo=PROB_FOGO):
    r = random.random()
    if r < prob_bomba:
        return 3
    elif r < prob_bomba + prob_fogo:
        return 4
    return 0

def desenhar_hud(pontos, tempo_restante):
    pygame.draw.rect(screen, (30, 30, 30), (0, 0, WIDTH, HUD_HEIGHT))
    tempo_txt = font.render(f"Tempo: {int(tempo_restante)}s", True, (255, 255, 255))
    screen.blit(tempo_txt, (10, 10))
    for i, p in enumerate(players):
        cor = COLOR_PLAYERS[i]
        nome_jogador = ""
        if i==0:
            nome_jogador = p1
        if i==1:
            nome_jogador = p2    
        if i==2:
            nome_jogador = p3
        if i==3:
            nome_jogador = p4    
        if i==4:
            nome_jogador = p5  
        txt = font.render(f"{nome_jogador}: {pontos[i]}", True, cor if p.ativo else (100, 100, 100))
        screen.blit(txt, (150 + i * 120, 10))

class Bomba:
    def __init__(self, x, y, tempo_explosao, nivel=1, dono=None):
        self.x = x
        self.y = y
        self.tempo_explosao = tempo_explosao
        self.explodida = False
        self.nivel = nivel
        self.tempo_fogo = 0
        self.fogo = []
        self.dono = dono

         # Para animação
        self.anim_frame = 0
        self.anim_timer = 0
        self.anim_interval = 0.2  # segundos por frame

    def atualizar(self, delta_time):
        if not self.explodida:
            self.tempo_explosao -= delta_time

            # Atualiza frame da bomba
            self.anim_timer += delta_time
            if self.anim_timer >= self.anim_interval:
                self.anim_timer = 0
                self.anim_frame = (self.anim_frame + 1) % len(bomb_frames)

            if self.tempo_explosao <= 0:
                self.explodir()
        else:
            self.tempo_fogo -= delta_time

    def explodir(self):
        self.explodida = True
        self.tempo_fogo = TEMPO_FOGO
        self.fogo.append((self.x, self.y))
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            for i in range(1 + (self.nivel - 1) * 2):
                nx = self.x + dx * (i + 1)
                ny = self.y + dy * (i + 1)
                if not (0 <= nx < COLS and 0 <= ny < ROWS):
                    break
                if mapa[ny][nx] == 2:
                    break
                self.fogo.append((nx, ny))
                if mapa[ny][nx] == 1:
                    break
        for fx, fy in self.fogo:
            for outra in bombas:
                if not outra.explodida and (outra.x, outra.y) == (fx, fy):
                    outra.explodir()

class Player:
    def __init__(self, x, y, tipo, time, cor_id, ativo=True, ia_fn=None):
        self.grid_x = x
        self.grid_y = y
        self.pixel_x = x * TILE_SIZE
        self.pixel_y = y * TILE_SIZE
        self.dest_x = self.pixel_x
        self.dest_y = self.pixel_y
        self.tipo = tipo
        self.time = time
        
        # Carrega spritesheet do jogador
        sprite_sheet = pygame.image.load(f"player{cor_id+1}.png").convert_alpha()
        self.frames = []
        for i in range(12):
            frame = sprite_sheet.subsurface(pygame.Rect(i * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE))
            self.frames.append(pygame.transform.scale(frame, (TILE_SIZE, TILE_SIZE)))

        self.anim_frame = 0
        self.anim_timer = 0
        self.anim_interval = 0.15
        self.ultima_direcao = "baixo"


        self.movendo = False
        self.tempo_mov = 0
        self.velocidade = TILE_SIZE / TEMPO_MOVIMENTO
        self.max_bombas = 1
        self.bombas = []
        self.bomba_nivel = 1
        self.ativo = ativo
        self.ia_fn = ia_fn

    def get_acao(self, teclas):
        if vencedor_final is not None or not self.ativo: 
            return "parado"
        if self.tipo == "user":
            return get_acao_usuario(teclas)

        elif self.tipo == "ia" and self.ia_fn:
            self_state = {
                'grid_x': self.grid_x,
                'grid_y': self.grid_y,
                'max_bombas': self.max_bombas,
                'bombas_ativas': len(self.bombas),
                'bomba_nivel': self.bomba_nivel,
                'ativo': self.ativo,
            }

            hud_info = {
                'tile_size': TILE_SIZE,
                'rows': ROWS,
                'cols': COLS,
                'tempo_movimento': TEMPO_MOVIMENTO,
                'tempo_explosao': TEMPO_EXPLOSAO,
                'tempo_fogo': TEMPO_FOGO,
                'max_bombas': MAX_BOMBAS,
            }

            chks_player = gerar_checksum_dados(self)
            chks_mapa = gerar_checksum_dados(mapa)
            chks_players = gerar_checksum_dados(players)
            chks_bombas = gerar_checksum_dados(bombas)
            chks_pontos = gerar_checksum_dados(pontos)

            acao = self.ia_fn(self, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)

            cheat = False
            if gerar_checksum_dados(self) != chks_player:
                print(f"[ALERTA] IA do jogador {players.index(self)+1} modificou o próprio objeto.")
                cheat = True
            if gerar_checksum_dados(mapa) != chks_mapa:
                print(f"[ALERTA] IA do jogador {players.index(self)+1} modificou o mapa.")
                cheat = True
            if gerar_checksum_dados(players) != chks_players:
                print(f"[ALERTA] IA do jogador {players.index(self)+1} modificou a lista de jogadores.")
                cheat = True
            if gerar_checksum_dados(bombas) != chks_bombas:
                print(f"[ALERTA] IA do jogador {players.index(self)+1} modificou as bombas.")
                cheat = True
            if gerar_checksum_dados(pontos) != chks_pontos:
                print(f"[ALERTA] IA do jogador {players.index(self)+1} modificou os pontos.")
                cheat = True

            if cheat:
                for p in players:
                    p.tipo = "cheat"
                self.max_bombas = 10
                return "bomba"
            return acao

        return "parado"

    def iniciar_movimento(self, dx, dy):
        self.grid_x += dx
        self.grid_y += dy

        if dy == -1:
            self.ultima_direcao = "cima"
        elif dy == 1:
            self.ultima_direcao = "baixo"
        elif dx == -1:
            self.ultima_direcao = "esquerda"
        elif dx == 1:
            self.ultima_direcao = "direita"

        self.dest_x = self.grid_x * TILE_SIZE
        self.dest_y = self.grid_y * TILE_SIZE
        self.movendo = True
        self.tempo_mov = 0

    def atualizar(self, delta_time):
        if self.movendo:
            self.tempo_mov += delta_time
            if self.tempo_mov >= TEMPO_MOVIMENTO:
                self.pixel_x = self.dest_x
                self.pixel_y = self.dest_y
                self.movendo = False
                tipo = mapa[self.grid_y][self.grid_x]
                if tipo == 3:
                    if self.max_bombas < MAX_BOMBAS:
                        self.max_bombas += 1
                    mapa[self.grid_y][self.grid_x] = 0
                    pontos[players.index(self)] += PONTOS_POWERUP_COLETADO
                elif tipo == 4:
                    if self.bomba_nivel < 4:
                        self.bomba_nivel += 1
                    mapa[self.grid_y][self.grid_x] = 0
                    pontos[players.index(self)] += PONTOS_POWERUP_COLETADO
            else:
                t = self.tempo_mov / TEMPO_MOVIMENTO
                self.pixel_x = (1 - t) * self.pixel_x + t * self.dest_x
                self.pixel_y = (1 - t) * self.pixel_y + t * self.dest_y

def get_acao_usuario(teclas):
    if teclas[pygame.K_SPACE]: return "bomba"
    elif teclas[pygame.K_UP]: return "cima"
    elif teclas[pygame.K_DOWN]: return "baixo"
    elif teclas[pygame.K_LEFT]: return "esquerda"
    elif teclas[pygame.K_RIGHT]: return "direita"
    return "parado"

# def desenhar_mapa():
#     for y in range(ROWS):
#         for x in range(COLS):
#             val = mapa[y][x]
#             pygame.draw.rect(screen, COLOR_GRASS, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
#             if val == 0:
#                 pygame.draw.rect(screen, COLOR_FLOOR, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
#             elif val == 1:
#                 pygame.draw.rect(screen, COLOR_BREAKABLE, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
#             elif val == 2:
#                 pygame.draw.rect(screen, COLOR_INDESTRUCTIBLE, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
#             elif val == 3:
#                 screen.blit(powerup_bomba_img, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT))
#             elif val == 4:
#                 screen.blit(powerup_fogo_img, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT))
#             pygame.draw.rect(screen, COLOR_BG, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE), 1)

def desenhar_mapa():
    for y in range(ROWS):
        for x in range(COLS):
            val = mapa[y][x]
            pygame.draw.rect(screen, COLOR_GRASS, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
            if val == 0:
                pygame.draw.rect(screen, COLOR_FLOOR, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
            elif val == 1:
                screen.blit(bloco_img, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT))  # ⬅ substituição aqui
            elif val == 2:
                pygame.draw.rect(screen, COLOR_INDESTRUCTIBLE, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE))
            elif val == 3:
                screen.blit(powerup_bomba_img, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT))
            elif val == 4:
                screen.blit(powerup_fogo_img, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT))
            pygame.draw.rect(screen, COLOR_BG, (x * TILE_SIZE, y * TILE_SIZE + HUD_HEIGHT, TILE_SIZE, TILE_SIZE), 1)


def desenhar_jogadores(players):
    for p in players:
        if not p.ativo:
            continue

        # Atualiza animação
        if p.movendo:
            p.anim_timer += 1
            if p.anim_timer >= 8:
                p.anim_timer = 0
                p.anim_frame = (p.anim_frame + 1) % 3
        else:
            p.anim_frame = 0  # parado = 1º frame da direção

        direcao_idx = {"cima": 0, "direita": 1, "baixo": 2, "esquerda": 3}
        base = direcao_idx[p.ultima_direcao] * 3
        frame = p.frames[base + p.anim_frame]

        screen.blit(frame, (p.pixel_x, p.pixel_y + HUD_HEIGHT))


def desenhar_bombas(bombas):
    for b in bombas:
        if not b.explodida:
            frame = bomb_frames[b.anim_frame]
            screen.blit(frame, (b.x * TILE_SIZE, b.y * TILE_SIZE + HUD_HEIGHT))
        else:
            for fx, fy in b.fogo:
                pos_x = fx * TILE_SIZE
                pos_y = fy * TILE_SIZE + HUD_HEIGHT

                if (fx, fy) == (b.x, b.y):
                    # Centro da explosão
                    frame = explosao_frames[1]
                elif fx == b.x:
                    if fy < b.y:
                        # Cima
                        if (fx, fy - 1) not in b.fogo:
                            frame = explosao_frames[0]  # ponta cima
                        else:
                            frame = explosao_frames[3]  # corpo vertical
                    elif fy > b.y:
                        # Baixo
                        if (fx, fy + 1) not in b.fogo:
                            frame = pygame.transform.rotate(explosao_frames[0], 180)  # ponta baixo
                        else:
                            frame = explosao_frames[3]  # corpo vertical
                elif fy == b.y:
                    if fx > b.x:
                        # Direita
                        if (fx + 1, fy) not in b.fogo:
                            frame = explosao_frames[2]  # ponta direita
                        else:
                            frame = explosao_frames[4]  # corpo horizontal
                    elif fx < b.x:
                        # Esquerda
                        if (fx - 1, fy) not in b.fogo:
                            frame = pygame.transform.flip(explosao_frames[2], True, False)  # ponta esquerda
                        else:
                            frame = explosao_frames[4]  # corpo horizontal
                else:
                    # fallback para evitar crash (ex: posição inválida)
                    frame = explosao_frames[1]

                screen.blit(pygame.transform.scale(frame, (TILE_SIZE, TILE_SIZE)), (pos_x, pos_y))


p1 = "ia_jogador1"
p2 = "ia_jogador2"
p3 = "ia_jogador3"
p4 = "ia_jogador4"
p5 = "ia_jogador1"

ia_1 = importlib.import_module(p1).decidir_acao
ia_2 = importlib.import_module(p2).decidir_acao
ia_3 = importlib.import_module(p3).decidir_acao
ia_4 = importlib.import_module(p4).decidir_acao
ia_5 = importlib.import_module(p5).decidir_acao

players = [
    # params: psX, psY, tipo, time, cor, ativo?, ia
    Player(0, 0, "ia", 0, 0, ativo=True, ia_fn=ia_1),
    Player(12, 0, "ia", 1, 1, ativo=True, ia_fn=ia_2),
    Player(0, 10, "ia", 1, 2, ativo=True, ia_fn=ia_3),
    Player(12, 10, "ia", 0, 3, ativo=True, ia_fn=ia_4),
    # Player(6, 5, "ia", 1, 4, ativo=True, ia_fn=ia_5),
]

bombas = []
pontos = [0, 0, 0, 0, 0]
tempo_restante = TEMPO_PARTIDA
vencedor_final = None
mensagem_vitoria = None

clock = pygame.time.Clock()
fullscreen = False

while True:
    delta = clock.tick(60) / 1000 
    teclas = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            fullscreen = not fullscreen
            if fullscreen:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))

    if vencedor_final is None:
        tempo_restante -= delta
        if tempo_restante <= 0 and vencedor_final is None:
            tempo_restante = 0
            vencedor_final = pontos.index(max(pontos))
            nome_vencedor = ""
            if vencedor_final==0:
                nome_vencedor = p1
            if vencedor_final==1:
                nome_vencedor = p2
            if vencedor_final==2:
                nome_vencedor = p3
            if vencedor_final==3:
                nome_vencedor = p4
            if vencedor_final==4:
                nome_vencedor = p5
            mensagem_vitoria = f"Tempo esgotado! {nome_vencedor} venceu!"

    for p in players:
        if not p.ativo:
            continue
        p.atualizar(delta)
        if not p.movendo:
            acao = p.get_acao(teclas)
            dx, dy = 0, 0
            if acao == "cima": dy = -1
            elif acao == "baixo": dy = 1
            elif acao == "esquerda": dx = -1
            elif acao == "direita": dx = 1
            elif acao == "bomba":
                if len(p.bombas) < p.max_bombas:
                    existe_bomba = any(b.x == p.grid_x and b.y == p.grid_y and not b.explodida for b in bombas)
                    if not existe_bomba:
                        nova = Bomba(p.grid_x, p.grid_y, TEMPO_EXPLOSAO, p.bomba_nivel, p)
                        bombas.append(nova)
                        p.bombas.append(nova)
            nx, ny = p.grid_x + dx, p.grid_y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS and mapa[ny][nx] in [0, 3, 4]:
                existe_bomba = any(b.x == nx and b.y == ny and not b.explodida for b in bombas)
                bomba_sob_player = any(b.x == p.grid_x and b.y == p.grid_y and not b.explodida for b in p.bombas)
                if not existe_bomba or (nx == p.grid_x and ny == p.grid_y and bomba_sob_player):
                    if (dx != 0 or dy != 0):
                        p.iniciar_movimento(dx, dy)

    if vencedor_final is None:
        for b in bombas[:]:
            b.atualizar(delta)
            if b.explodida and b.tempo_fogo <= 0:
                for fx, fy in b.fogo:
                    if mapa[fy][fx] == 1:
                        mapa[fy][fx] = criar_powerup(fx, fy)
                        if b.dono:
                            pontos[players.index(b.dono)] += PONTOS_BLOCO
                    elif mapa[fy][fx] in [3, 4]:
                        mapa[fy][fx] = 0
                        if b.dono:
                            pontos[players.index(b.dono)] += PONTOS_POWERUP_DESTRUIDO
                bombas.remove(b)
                if b.dono and b in b.dono.bombas:
                    b.dono.bombas.remove(b)

        for b in bombas:
            if b.explodida and b.tempo_fogo > 0:
                for p in players:
                    if p.ativo and (p.grid_x, p.grid_y) in b.fogo:
                        p.ativo = False
                        if b.dono == p:
                            pontos[players.index(p)] -= PONTOS_MATAR_JOGADOR
                            if pontos[players.index(p)]<0:
                                pontos[players.index(p)]=0;

                        else:
                            pontos[players.index(b.dono)] += PONTOS_MATAR_JOGADOR
                        print(f"Jogador {players.index(p)+1} morreu!")

    if vencedor_final is None:
        vivos = [p for p in players if p.ativo]
        if len(vivos) == 1:
            vencedor_final = players.index(vivos[0])
            pontos[vencedor_final] += PONTOS_VITORIA
            if vencedor_final==0:
                nome_vencedor = p1
            if vencedor_final==1:
                nome_vencedor = p2
            if vencedor_final==2:
                nome_vencedor = p3
            if vencedor_final==3:
                nome_vencedor = p4
            if vencedor_final==4:
                nome_vencedor = p5
            mensagem_vitoria = f"Jogador {nome_vencedor} VENCEU!"

    screen.fill(COLOR_BG)
    desenhar_hud(pontos, tempo_restante)
    desenhar_mapa()
    desenhar_bombas(bombas)
    desenhar_jogadores(players)
    if mensagem_vitoria:
        texto = font_vitoria.render(mensagem_vitoria, True, (255, 255, 255))
        rect = texto.get_rect(center=(WIDTH // 2, HEIGHT // 2 + HUD_HEIGHT // 2))
        screen.blit(texto, rect)
    pygame.display.flip()

    if vencedor_final is not None:
        # Apenas desenha a tela congelada com a mensagem
        # screen.fill(COLOR_BG)
        # desenhar_hud(pontos, tempo_restante)
        # desenhar_mapa()
        # desenhar_bombas(bombas)
        # desenhar_jogadores(players)
        if mensagem_vitoria:
            texto = font_vitoria.render(mensagem_vitoria, True, (255, 255, 255))
            rect = texto.get_rect(center=(WIDTH // 2, HEIGHT // 2 + HUD_HEIGHT // 2))
            screen.blit(texto, rect)
        pygame.display.flip()
        continue  # pula o restante do loop para congelar o jogo