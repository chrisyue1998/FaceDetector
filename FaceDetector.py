import cv2

running = True

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

OUTPUT_SIZE_WIDTH = 640
OUTPUT_SIZE_HEIGHT = 360

capture = cv2.VideoCapture(0)

cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("base-image", 0, 100)
cv2.moveWindow("result-image", 400, 100)

cv2.startWindowThread()

rectangleColor = (0, 165, 255)

while running:
    rc, fullSizeBaseImage = capture.read()
    baseImage = cv2.resize(fullSizeBaseImage, (320, 180))

    pressedKey = cv2.waitKey(2)
    if pressedKey == ord('Q'):
        running = False
        cv2.destroyAllWindows()
        exit(0)

    resultImage = baseImage.copy()

    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0

    for (_x, _y, _w, _h) in faces:
        if _w * _h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w * h

        if maxArea > 0:
            cv2.rectangle(resultImage, (x - 10, y - 20), (x + w + 10, y + h + 20), rectangleColor, 2)

    largeResult = cv2.resize(resultImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

    cv2.imshow("base-image", baseImage)
    cv2.imshow("result-image", largeResult)
