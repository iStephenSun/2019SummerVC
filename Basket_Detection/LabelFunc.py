import cv2
import numpy as np


def getHoopPosition(fn_video):
    hoopPos = np.zeros((2, 2), np.int)

    def onMouse(event, x, y, flags, param):
        # 鼠标左键点击，记录左上角点
        if event == cv2.EVENT_LBUTTONDOWN:
            hoopPos[0, :] = x, y  # hoopPos row0
        # 鼠标左键抬起，记录右下角点
        elif event == cv2.EVENT_LBUTTONUP:
            hoopPos[1, :] = x, y
            cv2.destroyWindow('image')

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', onMouse)
    # load
    cap = cv2.VideoCapture(fn_video)
    success, frame = cap.read()

    cv2.imshow('image', frame)
    cv2.waitKey(0)

    cv2.rectangle(frame, (hoopPos[0, 0], hoopPos[0, 1]), (hoopPos[1, 0], hoopPos[1, 1]), (0, 255, 0), 1)
    cv2.imshow('image', frame)
    cv2.waitKey(1800)

    # release
    cap.release()
    cv2.destroyAllWindows()

    print('The position of hoop is:')
    print(hoopPos)
    return hoopPos


def nothing(x):
    pass


def labelGoalFrames(fn_video, hoopPos, fn_annotation):
    name = 'Video'
    waitTime = 40
    # index
    frame_index = 0
    cur_frame = []

    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 800, 600)
    cap = cv2.VideoCapture(fn_video)

    # 显示视频帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("FPS of the video: ", fps)
    # 显示视频总帧数
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames: ", frame_counts)

    pos = 0
    cv2.createTrackbar('frame', name, 0, frame_counts, nothing)

    while True:

        if frame_index == pos:
            frame_index = frame_index + 1
            cv2.setTrackbarPos('frame', name, frame_index)
        else:
            pos = cv2.getTrackbarPos('frame', name)
            frame_index = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        success, frame = cap.read()

        cv2.rectangle(frame, (hoopPos[0, 0], hoopPos[0, 1]), (hoopPos[1, 0], hoopPos[1, 1]), (0, 255, 0), 1)
        cv2.imshow(name, frame)

        k = cv2.waitKey(waitTime)

        # esc  退出
        if k == 27:
            break
        # response
        elif k == 32:  # space
            while (1):
                cur_frame.append(frame_index)
                flag = cv2.waitKey(0)
                if flag == 32:
                    break

    # release
    cap.release()
    cv2.destroyAllWindows()

    print(cur_frame)
    # write file
    f = open(fn_annotation, 'a')
    for i in cur_frame:
        f.write(str(i) + '\n')
    f.close()


def cropHoop(fn_video, hoopPos, fn_annotation, crop_dir_pos, crop_dir_neg):
    pos_frame = []
    with open(fn_annotation) as fdata:
        while True:
            line = fdata.readline()
            if not line:
                break
            pos_frame.append(int(line))

    print(pos_frame)

    cap = cv2.VideoCapture(fn_video)
    # 显示视频帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("FPS of the video: ", fps)
    # 显示视频总帧数
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames: ", frame_counts)

    # index
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if frame_index in pos_frame:
                # pick out gray hoop
                frameCroped = frame[hoopPos[0, 1]:hoopPos[1, 1], hoopPos[0, 0]:hoopPos[1, 0]]
                imgGrey = cv2.cvtColor(frameCroped, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(imgGrey, (40, 40), cv2.INTER_LINEAR)
                # crop_file name
                crop_pic_name = crop_dir_pos + "positive" + str(frame_index) + ".jpg"
                cv2.imwrite(crop_pic_name, img)
                print("saving positive", frame_index, " pictures")

            if frame_index not in pos_frame:
                # pick out gray hoop
                frameCroped = frame[hoopPos[0, 1]:hoopPos[1, 1], hoopPos[0, 0]:hoopPos[1, 0]]
                imgGrey = cv2.cvtColor(frameCroped, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(imgGrey, (40, 40), cv2.INTER_LINEAR)
                # crop_file name
                crop_pic_name = crop_dir_neg + "negative" + str(frame_index) + ".jpg"
                cv2.imwrite(crop_pic_name, img)
                print("saving negative", frame_index, " pictures")

            # index+1
            frame_index += 1

            if frame_index == frame_counts:
                print("all the pictures have been saved correctly!")
                break

    cap.release()
    cv2.destroyAllWindows()
