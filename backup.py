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

# Carregar o modelo YOLO
# model = YOLO("best.pt")
model = YOLO("yolov8s_playing_cards.pt")

# Função para detectar cartas e calcular o total
def detectar_e_somar_cartas(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Fazer a previsão
        predicoes = model.predict(source=frame, show=False)

        # Inicializa um conjunto para cartas detectadas no quadro atual
        cartas_detectadas_atual = set()

        # Acessa as detecções
        for result in predicoes:
            for box in result.boxes:
                classe_detectada = result.names[int(box.cls)]  # Obter o nome da classe
                
                # Adicionar carta se estiver no conjunto de valores
                if classe_detectada in valores_cartas:
                    cartas_detectadas_atual.add(classe_detectada)

        # Calcular a soma para o quadro atual
        soma_total_atual = sum(valores_cartas[carta] for carta in cartas_detectadas_atual)

        # Mostrar o total na tela
        cv2.putText(frame, f'Total: {soma_total_atual}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if soma_total_atual > 21: cv2.putText(frame, 'Você perdeu', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Executar a função
#detectar_e_somar_cartas("D:\\Proj\\Python\\yolo_11_custom\\1.mp4")  # Substitua pelo caminho do seu vídeo
detectar_e_somar_cartas(0)
