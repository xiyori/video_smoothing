import cv2
import average as method

# cap = cv2.VideoCapture("test1.mp4")
# cap = cv2.VideoCapture("test2.MOD")
# cap = cv2.VideoCapture("test3.MOD")
# cap = cv2.VideoCapture("test4.MOD")
cap = cv2.VideoCapture("test5.MOV")

method.stabilize(cap, 200)

cap.release()
cv2.destroyAllWindows()
