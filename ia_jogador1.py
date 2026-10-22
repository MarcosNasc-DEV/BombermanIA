import random

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]

def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    return random.choice(ACOES)
