from draw_hand import *
from Utils.Check import Detect_joint

# --- 메인 코드 실행 ---
def debug_test(idx):
    
    video_file = fr'c:\Users\User\Desktop\원천데이터\REAL\WORD\01\NIA_SL_WORD15{idx :02d}_REAL01_F.mp4'  
    # video_file = r'c:\Users\User\Desktop\frame\output_0120.png'  

    original_data, dims = linear_joint(video_file)

 
    if original_data is not None:
        # --- 핵심 추가 부분: 손실 프레임 계산 및 출력 ---
        total_frames = len(original_data)
        left_hand_lost = original_data['left_landmark_x_0'].isna().sum()
        right_hand_lost = original_data['right_landmark_x_0'].isna().sum()
        # -----------------------------------------------

        debug_data = (idx , left_hand_lost , right_hand_lost)
        Detect_joint(original_data , is_debug= True , debug_data = debug_data)    
        # visualize_debug(video_file, original_data, interpolated_data, dims, loss_info_tuple)
        # create_keypoint_video(f'debug//test{idx}.mp4' , original_data, interpolated_data, dims , is_debug = True)
        
from tqdm import tqdm 
        
for idx in tqdm(range(1,21) , '진행중'):
    debug_test(idx)
    