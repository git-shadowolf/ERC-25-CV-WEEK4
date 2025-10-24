import cv2
import mediapipe as mp
import numpy as np 



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#the only diff is we only track  hand to avoid confusion
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              
    min_detection_confidence=0.7  
)



#prev position of finger or basically finger position in prev frame
prev_x, prev_y = None, None

#start with red 
draw_color = (0, 0, 255)#BGR for openCV
line_thickness = 5
eraser_mode = False



print("Starting webcam... Get your drawing finger ready!")
cap = cv2.VideoCapture(0)


success, test_frame = cap.read()
if not success:
    print("FATAL: Failed to get frame from webcam. Exiting.")
    exit()

#size of the frame 
h, w, _ = test_frame.shape

#blank canvas to draw on of same size as frame 
canvas = np.zeros((h, w, 3), dtype=np.uint8)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
      continue

    
    frame = cv2.flip(frame, 1)
    
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

   

    #found hand?
    if results.multi_hand_landmarks:
        #for 1 hand we put that [0]
        hand_landmarks = results.multi_hand_landmarks[0]
        
        #caring about tip of index finger only , i.e out 'brush'
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        #since landmark coords are normalized, gotta scale them to required coordinate system 
        curr_x = int(index_tip.x * w)
        curr_y = int(index_tip.y * h)
        
        
        
        #hand just entered frame 
        if prev_x is None:
            # so set prev pt to new/current pt.
            prev_x, prev_y = curr_x, curr_y
        
        #pen or eraser 
        if eraser_mode:
            color_to_draw = (0, 0, 0) #basically overwrites with black on canvas we had 
            thickness_to_draw = 20     
        else:
            color_to_draw = draw_color
            thickness_to_draw = line_thickness

        #draw line from prev pt to current pt 
        cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), color_to_draw, thickness_to_draw)
        
        #update IMP so that next pt connect to prev pt in next loop 
        prev_x, prev_y = curr_x, curr_y
        
    else:
        #lifted the pen so reset all (or we didnt find a hand basially )
        prev_x, prev_y = None, None

   

    #overlay canvas made onto og frame 
    result_frame = cv2.add(frame, canvas)
    
    #instructions onto the livefeed 
    cv2.putText(result_frame, "Controls:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, "'r', 'g', 'b' = Colors", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, "'e' = Eraser | 'c' = Clear", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, "'+' / '-' = Thickness", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, "'q' = Quit", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #display 
    cv2.imshow("CV Drawing Pad", result_frame)

    #key controls for all functions 
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break # Quit
    elif key == ord('r'):
        draw_color = (0, 0, 255)#R
        eraser_mode = False
    elif key == ord('g'):
        draw_color = (0, 255, 0)#G
        eraser_mode = False
    elif key == ord('b'):
        draw_color = (255, 0, 0)#B
        eraser_mode = False
    elif key == ord('e'):
        eraser_mode = not eraser_mode#ERASE
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)#clear entire canavs replace it with 0s
    elif key == ord('+') or key == ord('='):
        line_thickness = min(20, line_thickness + 2)
    elif key == ord('-'):
        line_thickness = max(1, line_thickness - 2)


print("Shutting down...")
cap.release()
cv2.destroyAllWindows()