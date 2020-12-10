import cv2
import numpy as np
import numpy.linalg as lin


def smooth_trajectory(cap: cv2.VideoCapture, amount: int) -> list:
    max_n = amount
    n_shift = [np.identity(3)]
    apply_warp = []
    prev_frame = None
    curr_ind = 0

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(frame, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30,
                                           blockSize=3)
        if prev_frame is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                             prev_pts, None)
            M, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
            n_shift.append(np.dot(M, n_shift[-1]))
            if len(n_shift) > max_n:
                inv = lin.inv(n_shift.pop(0))
                for i in range(len(n_shift)):
                    n_shift[i] = np.dot(n_shift[i], inv)
            prev_frame = frame
            if curr_ind >= max_n:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(np.average(np.array(n_shift[max_n // 2 - 1: max_n // 2 + 1]), 0))))
            elif curr_ind > max_n // 2:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(np.average(np.array(n_shift[curr_ind - max_n // 2 - 1: curr_ind - max_n // 2 + 1]), 0))))
            elif curr_ind == max_n // 2:
                apply_warp.append(np.average(np.array(n_shift), 0))
        else:
            prev_frame = frame
        curr_ind += 1
    for curr_ind in range(0, max_n // 2):
        inv = lin.inv(n_shift.pop(0))
        for i in range(len(n_shift)):
            n_shift[i] = np.dot(n_shift[i], inv)
        if max_n // 2 < len(n_shift):
            apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                     lin.inv(np.average(np.array(n_shift[max_n // 2 - 1: max_n // 2 + 1]), 0))))
        else:
            apply_warp.append(np.average(np.array(n_shift), 0))
    return apply_warp


def smooth_cool(cap: cv2.VideoCapture, amount: int) -> list:
    max_n = amount
    n_shift = [np.identity(3)]
    apply_warp = []
    prev_frame = None
    curr_ind = 0

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(frame, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30,
                                           blockSize=3)
        if prev_frame is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                             prev_pts, None)
            M, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
            n_shift.append(np.dot(M, n_shift[-1]))
            if len(n_shift) > max_n:
                inv = lin.inv(n_shift.pop(0))
                for i in range(len(n_shift)):
                    n_shift[i] = np.dot(n_shift[i], inv)
            prev_frame = frame
            if curr_ind >= max_n:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(n_shift[max_n // 2])))
            elif curr_ind >= max_n // 2:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(n_shift[curr_ind - max_n // 2])))
        else:
            prev_frame = frame
        curr_ind += 1
    for curr_ind in range(0, max_n // 2):
        inv = lin.inv(n_shift.pop(0))
        for i in range(len(n_shift)):
            n_shift[i] = np.dot(n_shift[i], inv)
        if max_n // 2 < len(n_shift):
            apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                     lin.inv(n_shift[max_n // 2])))
        else:
            apply_warp.append(np.average(np.array(n_shift), 0))
    return apply_warp


def stabilize(cap: cv2.VideoCapture, amount: int) -> None:
    print("Processing data...")
    apply_warp = smooth_cool(cap, amount - amount % 2)
    print("Writing to disk...")
    cap.set(2, 0)
    curr_ind = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break
        frame_color = cv2.warpPerspective(frame_color, apply_warp[curr_ind], (w, h))
        curr_ind += 1

        out.write(frame_color)

        cv2.imshow('frame', frame_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    print("Done")
