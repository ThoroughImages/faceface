import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
img = cv2.imread("logo.png")


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.

    Ref: 
    https://stackoverflow.com/questions/14063070/
    overlay-a-smaller-image-on-a-larger-image-python-opencv/
    14102014#14102014
    """

    x, y = pos
    x = int(x)
    y = int(y)

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    for c in range(channels):
        img[y1: y2, x1: x2, c] = (alpha_mask * img_overlay[y1o: y2o, x1o: x2o, c] +
                                  (1.0 - alpha_mask) * img[y1: y2, x1: x2, c])
    return img


while True:
    # Capture frames from WebCamera
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 254), 5)

    # Full screen display
    frame = overlay_image_alpha(frame, img[:, :, 0: 3], (1, 1), 1)
    cv2.namedWindow("FaceFace", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("FaceFace", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('FaceFace', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleaning up
video_capture.release()
cv2.destroyAllWindows()
