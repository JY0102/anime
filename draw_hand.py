import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from Utils.Spline import spline

def linear_joint(video_path):
    """
    영상에서 양손의 '모든' 랜드마크를 추출하고, 누락된 프레임을 찾아 보간하는 함수.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  
        min_detection_confidence=0.9,
        min_tracking_confidence=0.5        
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: {video_path} 파일을 열 수 없습니다.")
        return None, None, (0, 0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_landmarks_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        frame_landmarks = {f'{side}_{h_type}_{i}': np.nan 
                           for side in ['left', 'right'] 
                           for h_type in ['landmark_x', 'landmark_y'] 
                           for i in range(21)}

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                hand_side = 'left' if handedness == 'Left' else 'right'
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    frame_landmarks[f'{hand_side}_landmark_x_{i}'] = landmark.x
                    frame_landmarks[f'{hand_side}_landmark_y_{i}'] = landmark.y
        
        all_landmarks_data.append(frame_landmarks)
    
    cap.release()
    hands.close()
    
    original_df = pd.DataFrame(all_landmarks_data)
    
    return original_df, (frame_width, frame_height)

def draw_hand(frame, landmark_row, original_row, hand_side, frame_dims , is_debug = False):
    """한 손의 랜드마크와 연결선을 그리는 보조 함수."""
    mp_hands = mp.solutions.hands
    frame_width, frame_height = frame_dims

    is_original = pd.notna(original_row[f'{hand_side}_landmark_x_0'])
    up_pos = 0
    
    if is_debug:
        dot_color = (255, 0, 0) if is_original else (0, 0, 255)
        line_color = (0, 255, 0) if is_original else (0, 255, 255)  
    else:
        dot_color = (255, 0, 0)
        line_color = (0, 255, 0)
        up_pos = 0.1

    points = []
    for i in range(21):
        x = landmark_row[f'{hand_side}_landmark_x_{i}']
        y = landmark_row[f'{hand_side}_landmark_y_{i}'] - up_pos
        if pd.notna(x) and pd.notna(y):
            points.append((int(x * frame_width), int(y * frame_height)))
        else:
            points.append(None)
    
    if points:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection[0], connection[1]
            if points[start_idx] and points[end_idx]:
                cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
        for point in points:
            if point:
                cv2.circle(frame, point, 5, dot_color, -1)
            
def visualize_debug(video_path, original_df, interpolated_df, frame_dimensions, loss_info):
    """
    보간된 양손의 결과와 프레임 손실 정보를 영상 위에 시각화하는 함수.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        original_row = original_df.loc[frame_idx]
        interpolated_row = interpolated_df.loc[frame_idx]
        
        draw_hand(frame, interpolated_row, original_row, 'left', frame_dimensions , is_debug=True)
        draw_hand(frame, interpolated_row, original_row, 'right', frame_dimensions , is_debug=True)

        # --- 핵심 추가 부분: 영상에 손실 프레임 정보 표시 ---
        total_frames, left_lost, right_lost = loss_info
        text_left = f"Left Lost: {left_lost}/{total_frames} ({left_lost/total_frames:.1%})"
        text_right = f"Right Lost: {right_lost}/{total_frames} ({right_lost/total_frames:.1%})"
        
        cv2.putText(frame, text_left, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, text_right, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # ---------------------------------------------------

        cv2.imshow('Both Hands Interpolation Visualization', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def create_keypoint_video(output_filename, original_df, interpolated_df, frame_dimensions, fps=30 , is_debug = False):
    """
    키포인트 애니메이션을 시각화하고 MP4 파일로 저장하는 함수.
    """
    frame_width, frame_height = frame_dimensions

    # --- 핵심 추가 부분: VideoWriter 객체 생성 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 코덱
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    # ---------------------------------------------

    
    for frame_idx in range(len(interpolated_df)):
        black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        original_row = original_df.loc[frame_idx]
        interpolated_row = interpolated_df.loc[frame_idx]
        
        draw_hand(black_canvas, interpolated_row, original_row, 'left', frame_dimensions , is_debug)
        draw_hand(black_canvas, interpolated_row, original_row, 'right', frame_dimensions , is_debug) 

        # cv2.imshow('Keypoint-Only Visualization', black_canvas)
        
        # --- 핵심 추가 부분: 프레임을 비디오 파일에 쓰기 ---
        video_writer.write(black_canvas)
        # ------------------------------------------------

    # --- 핵심 추가 부분: 자원 해제 ---
    video_writer.release()
    cv2.destroyAllWindows()
    # -----------------------------------
    

