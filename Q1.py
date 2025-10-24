
#import required libraries 
import cv2  #for webcam input and doing various things on this webcam feed. 
import mediapipe as mp #the ML part for identifying hands 

#setting up the mediapipe tools 
mp_hands = mp.solutions.hands #the model that recognises the Hand using, an instance of it is created and stored
mp_drawing = mp.solutions.drawing_utils #for drawing the skeleton, so we dont have to join the 21 dots by ourselves.

#the actual hand detection object is created and we pass it certain settings here
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5 #model consideres the object and hand only if its more than 50% sure.
)



print("Starting your webcam...")
#starts the webcam and stores it in cap.
cap = cv2.VideoCapture(0)

#infinite loop that keeps on capturing frames from webcam till u quit or cam closed
while cap.isOpened():
    
    #stores frame and sucess part tells if the frame captured or not.
    success, frame = cap.read()
    
   #checks if the frame captured or not(what if ur webcam malfunction, so exit) 
    if not success:
      print("Warning: Ignoring empty camera frame.")
      continue


    #actual work
    
    frame = cv2.flip(frame, 1) #to make the output we see more natural (mirror effect reversed basically)

    #media pipe need RGB but openCV captures in BGR so just converted them
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   #final result of ML model working is stored here, main part of everything basically.
    results = hands.process(rgb_frame)

 
   #draw if mediapipe found hands 
    if results.multi_hand_landmarks:
        
        #looping thorugh all hands it found in frame and drawing them.
        for hand_landmarks in results.multi_hand_landmarks:
            
            #direct func to draw/connect all 21 pts
            mp_drawing.draw_landmarks(
                frame,                  #what to draw on 
                hand_landmarks,         #the 21 pts to connect 
                mp_hands.HAND_CONNECTIONS #the lines that connect 21 pts 
            )

    #show the result 
    cv2.imshow("My Hand Tracker (Press 'q' to quit)", frame)

    #openCV waits for 1ms to see if a key has been pressed, if not keep going 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#exit 
print("Shutting down...")
cap.release() #close webcam 
cv2.destroyAllWindows()#close all windows (webcam window here)