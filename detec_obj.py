import cv2
import torch
from ultralytics import YOLO

# Verificar se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Carregar o modelo YOLO e movê-lo para a GPU
try:
    model = YOLO('yolov8s.pt').to(device)
    print("Modelo YOLO carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")

def detectar_objetos(results):
    objetos_detectados = []

    for result in results:
        for caixa, classe in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, caixa)

            # Obter o nome da classe detectada
            nome_classe = model.names[int(classe)]
            objetos_detectados.append(((x1, y1, x2, y2), nome_classe))

    print("Objetos detectados:", objetos_detectados)  # Print para depuração
    return objetos_detectados

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()
else:
    print("Câmera aberta com sucesso.")

cv2.namedWindow("Detecção de Objetos", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame")
            break

        # Redimensionar o frame para uma resolução compatível com YOLO, como 640x640
        frame_resized = cv2.resize(frame, (640, 640))

        # Converta o frame para um tensor, normalize dividindo por 255, e ajuste a dimensão para (1, 3, 640, 640)
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)

        # Fazer a detecção com o modelo na GPU
        results = model(frame_tensor)

        # Verificar e exibir todos os objetos detectados
        objetos_detectados = detectar_objetos(results)

        for (x1, y1, x2, y2), nome_objeto in objetos_detectados:
            # Ajustar as coordenadas do retângulo de volta para o tamanho original da janela
            x1, y1, x2, y2 = int(x1 * frame.shape[1] / 640), int(y1 * frame.shape[0] / 640), int(x2 * frame.shape[1] / 640), int(y2 * frame.shape[0] / 640)

            # Desenhar um retângulo ao redor do objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Adicionar o nome do objeto acima do retângulo
            cv2.putText(frame, f"{nome_objeto}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar o frame com as detecções
        cv2.imshow("Detecção de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupção pelo usuário detectada")

finally:
    cap.release()
    cv2.destroyAllWindows()
