from draw_hand import *

# --- 메인 코드 실행 ---
if __name__ == "__main__":
    
    video_file = r'c:\Users\User\Desktop\원천데이터\REAL\WORD\01\NIA_SL_WORD1507_REAL01_F.mp4'  
    # video_file = r'c:\Users\User\Desktop\frame\output_0120.png'  
    
    original_data, interpolated_data, dims = linear_joint(video_file)

    # from Utils.Check import Detect_joint
    # Detect_joint(original_data)


    if original_data is not None:
        # --- 핵심 추가 부분: 손실 프레임 계산 및 출력 ---
        total_frames = len(original_data)
        left_hand_lost = original_data['left_landmark_x_0'].isna().sum()
        right_hand_lost = original_data['right_landmark_x_0'].isna().sum()
        
        print("\n--- 프레임 손실 정보 ---")
        print(f"총 프레임: {total_frames}")
        print(f"왼손 감지 실패 프레임: {left_hand_lost} ({left_hand_lost/total_frames:.1%})")
        print(f"오른손 감지 실패 프레임: {right_hand_lost} ({right_hand_lost/total_frames:.1%})")
        # -----------------------------------------------

        loss_info_tuple = (total_frames, left_hand_lost, right_hand_lost)
        visualize_debug(video_file, original_data, interpolated_data, dims, loss_info_tuple)
        create_keypoint_video('test.mp4' , original_data, interpolated_data, dims , is_debug = True)