import cv2
import average as method

# cap = cv2.VideoCapture("test1.mp4")
# cap = cv2.VideoCapture(0)

method.stabilize(cap, 200)  # smoothing amount, larger value - lower frequency

cap.release()
cv2.destroyAllWindows()
