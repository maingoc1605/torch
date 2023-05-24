import cv2


link_rtsp = ["rtsp://admin:Ab123456@cam8ocd.cameraddns.net:556/Streaming/Channels/1",
"rtsp://admin:Hoangcau446@cam446dlt.cameraddns.net/:554/Streaming/Channels/1",
"rtsp://admin:Hoangcau79@cam179hc.cameraddns.net:557/Streaming/Channels/1",
"rtsp://admin:Abc123456@cam26hc.cameraddns.net:556/Streaming/Channels/1",
"rtsp://admin:123456Abcd@cam67hc.cameraddns.net:555/Streaming/Channels/1"]

cap = cv2.VideoCapture(link_rtsp[2])
width_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_stream = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or other fourcc codes
fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
out = cv2.VideoWriter(r"F:\Camera_ai\output\test_10.mp4", fourcc, fps_stream, (width_frame, height_frame))

while True:
    result, frame = cap.read()
    if result == True:
        out.write(frame)
        cv2.namedWindow("ALPR Test", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow('ALPR Test', frame)
        # print("Time for imshow: ", time.time() - start)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        continue

out.release()
cap.release()

