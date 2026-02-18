import cv2, os

name = "george"
save_path = f"C:/Users/georg/OneDrive/Desktop/centennial/2026 winter sem/comp385/facenet_mtcnn/dataset/train/{name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

for i in range(20):
    ret, frame = cap.read()
    cv2.imwrite(f"{save_path}/{i}.jpg", frame)
    cv2.imshow("capture", frame)
    cv2.waitKey(200)

cap.release()
cv2.destroyAllWindows()
