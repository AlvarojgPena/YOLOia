from ultralytics import YOLO
import cv2

# Valores das cartas
valores_cartas = {
    '2C': 2, '2D': 2, '2H': 2, '2S': 2,
    '3C': 3, '3D': 3, '3H': 3, '3S': 3,
    '4C': 4, '4D': 4, '4H': 4, '4S': 4,
    '5C': 5, '5D': 5, '5H': 5, '5S': 5,
    '6C': 6, '6D': 6, '6H': 6, '6S': 6,
    '7C': 7, '7D': 7, '7H': 7, '7S': 7,
    '8C': 8, '8D': 8, '8H': 8, '8S': 8,
    '9C': 9, '9D': 9, '9H': 9, '9S': 9,
    '10C': 10, '10D': 10, '10H': 10, '10S': 10,
    'JC': 10, 'JD': 10, 'JH': 10, 'JS': 10,
    'QC': 10, 'QD': 10, 'QH': 10, 'QS': 10,
    'KC': 10, 'KD': 10, 'KH': 10, 'KS': 10,
    'AC': 1, 'AD': 1, 'AH': 1, 'AS': 1
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
    time.sleep(5)
    cap.release()
    cv2.destroyAllWindows()

# Executar a função
#detectar_e_somar_cartas("D:\\Proj\\Python\\yolo_11_custom\\1.mp4")  # Substitua pelo caminho do seu vídeo
detectar_e_somar_cartas(0)
