import cv2

rtsp_url = "rtsp://admin:Haithy123@192.168.1.87:554/onvif1"

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame")
        break

    cv2.imshow("Yoosee Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()