import cv2
import numpy as np
import numpy.linalg as lin


def get_sub(prev_warp: list, current_warp: np.array, max_n: int) -> tuple:
    best_dist = float("+inf")
    best_warp_ind = -1
    target_warp = lin.inv(current_warp)
    for i in range(len(prev_warp)):
        new_dist = lin.norm(np.dot(prev_warp[max_n // 2], lin.inv(prev_warp[i])) - target_warp)
        if new_dist < best_dist:
            best_dist = new_dist
            best_warp_ind = i
    sub_warp = None
    if best_warp_ind != -1:
        sub_warp = np.dot(prev_warp[max_n // 2], lin.inv(prev_warp[best_warp_ind]))
    return best_warp_ind - max_n // 2, sub_warp


def smooth_trajectory(cap: cv2.VideoCapture, amount: int) -> tuple:
    max_n = amount
    n_shift = [np.identity(3)]
    apply_warp = []
    prev_frame = None
    curr_ind = 0
    sub_layers = [(0, None)]

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
                n_shift.pop(0)
                inv = lin.inv(n_shift[0])
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

            if curr_ind > max_n // 2:
                sub_layers.append(get_sub(n_shift, apply_warp[-1], max_n))
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
    return apply_warp, sub_layers


def smooth_cool(cap: cv2.VideoCapture, amount: int) -> tuple:
    n_shift = [np.identity(3)]
    apply_warp = []
    prev_frame = None
    curr_ind = 0
    sub_layers = [(0, None)]

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
            if len(n_shift) > amount:
                n_shift.pop(0)
                inv = lin.inv(n_shift[0])
                for i in range(len(n_shift)):
                    n_shift[i] = np.dot(n_shift[i], inv)
            prev_frame = frame
            if curr_ind >= amount:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(n_shift[amount // 2])))
            elif curr_ind >= amount // 2:
                apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                         lin.inv(n_shift[curr_ind - amount // 2])))

            if curr_ind > amount // 2:
                sub_layers.append(get_sub(n_shift, apply_warp[-1], amount))
        else:
            prev_frame = frame
        curr_ind += 1
    for curr_ind in range(0, amount // 2):
        n_shift.pop(0)
        inv = lin.inv(n_shift[0])
        for i in range(len(n_shift)):
            n_shift[i] = np.dot(n_shift[i], inv)
        if amount // 2 < len(n_shift):
            apply_warp.append(np.dot(np.average(np.array(n_shift), 0),
                                     lin.inv(n_shift[amount // 2])))
        else:
            apply_warp.append(np.average(np.array(n_shift), 0))
    return apply_warp, sub_layers


def stabilize(cap: cv2.VideoCapture, amount: int) -> None:
    amount = amount - amount % 2

    print("Processing data...")

    apply_warp, sub_layers = smooth_cool(cap, amount)

    print("Applying smoothing...")

    cap.set(2, 0)
    frames = []

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break
        frames.append(frame_color)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    print("Writing to disk...")

    for curr_ind in range(len(frames)):
        frame_color = cv2.warpPerspective(frames[curr_ind], apply_warp[curr_ind], (w, h))
        if sub_layers[curr_ind][1] is not None:
            mask = cv2.warpPerspective(np.full(frame_color.shape, 255, dtype=np.uint8),
                                       apply_warp[curr_ind], (w, h)) != 255
            np.putmask(frame_color, mask,
                       cv2.warpPerspective(frames[curr_ind + sub_layers[curr_ind][0]],
                                           np.dot(apply_warp[curr_ind],
                                                  sub_layers[curr_ind][1]), (w, h)))
        out.write(frame_color)
        cv2.imshow('frame', frame_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    print("Done")
