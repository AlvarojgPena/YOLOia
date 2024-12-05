from ultralytics import YOLO
import cv2

# Valores das cartas
valores_cartas = {
    '2c': 2, '2d': 2, '2h': 2, '2s': 2,
    '3c': 3, '3d': 3, '3h': 3, '3s': 3,
    '4c': 4, '4d': 4, '4h': 4, '4s': 4,
    '5c': 5, '5d': 5, '5h': 5, '5s': 5,
    '6c': 6, '6d': 6, '6h': 6, '6s': 6,
    '7c': 7, '7d': 7, '7h': 7, '7s': 7,
    '8c': 8, '8d': 8, '8h': 8, '8s': 8,
    '9c': 9, '9d': 9, '9h': 9, '9s': 9,
    '10c': 10, '10d': 10, '10h': 10, '10s': 10,
    'Jc': 10, 'Jd': 10, 'Jh': 10, 'Js': 10,
    'Qc': 10, 'Qd': 10, 'Qh': 10, 'Qs': 10,
    'Kc': 10, 'Kd': 10, 'Kh': 10, 'Ks': 10,
    'Ac': 11, 'Ad': 11, 'Ah': 11, 'As': 11
}

def somar_valores(cartas_detectadas):
    soma = 0
    for carta in cartas_detectadas:
        if carta in valores_cartas:
            soma += valores_cartas[carta]
    return soma

# Carregar o modelo YOLO
model = YOLO("best.pt")

# Fazer a previsão
predicoes = model.predict(source='1.jpg', show=True)

# Inicializa a lista de cartas detectadas
cartas_detectadas = set()

# Acessa as detecções
for result in predicoes:
    for box in result.boxes:
        classe_detectada = result.names[int(box.cls)]  # Obter o nome da classe
        if classe_detectada in valores_cartas:
            cartas_detectadas.add(classe_detectada)

# Calcular a soma das cartas detectadas
soma_total = somar_valores(cartas_detectadas)
print(f'Soma total das cartas detectadas: {soma_total}')
