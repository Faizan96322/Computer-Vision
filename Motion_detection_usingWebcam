import cv2, pandas, time
from datetime import datetime

static_back = None
current_motion = ["", ""]
Motionlist = []

# video caputre :
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    # initial motion is 0
    motion = 0

    # converting the input to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converting the grayscale to gaussian blur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back, gray)

    # if changes between staticback and gray is > 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # finding contours of moving objects
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            motion = 1
            continue

        if current_motion[0] == "" and motion == 1:
            current_motion[0] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        (x, y, w, h) = cv2.boundingRect(contour)
        # marking a rectangle around moving objects
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # append status of motion/ recording the end time of the motion
    if motion == 0 and current_motion[0] != "":
        current_motion[1] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        Motionlist.append(current_motion)
        current_motion = ["", ""]

    # Display the image in gray scale
    cv2.imshow("Gray Frame", gray)
    # Display the diff_frame
    cv2.imshow("difference frame", diff_frame)
    # Display the threshold frame
    cv2.imshow("threshold frame", thresh_frame)
    # Display the color frame
    cv2.imshow("color frame", frame)

    key = cv2.waitKey(1)
    # if q is pressed the recording will stop
    if key == ord('q'):
        # if something is still moving then the end time will be altered!
        if motion == 0 and current_motion[0] != "":
            current_motion[1] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            Motionlist.append(current_motion)
        break

for move in Motionlist[:30]:
    print(move)

video.release()

cv2.destroyAllWindows()

