import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
from tqdm import tqdm

from Utils.draw import *
from Utils.Spline import *
from Utils.Check import Detect_joint


def write_log(log_data, idx):
    sus, loss_info_tuple = log_data
    if not loss_info_tuple:
        total_frames = 'Error'
        left_lost = 'Error'
        right_lost = 'Error'
    else:
        total_frames, left_lost, right_lost = loss_info_tuple
    
    base_dir = 'output'
    path  = os.path.join(base_dir , 'out_log.txt')    
    os.makedirs(base_dir, exist_ok=True)
    
    if os.path.exists(path) and idx == 1:
        os.remove(path)
        
    with open(path, "a", encoding='utf-8') as file:
        
        file.write(f'{idx}번째 영상\n')
        file.write(f'프레임손실율(전체:왼손:오른손) : {total_frames} : {left_lost} : {right_lost}\n')
        file.write(f'영상 생성 : {sus}')
        file.write(f'\n\n')
                        
def check_video(video_path):
    
    _, file_extension = os.path.splitext(video_path)
    if not file_extension == '':
        return None
    video_names = []
    for video_name in sorted(os.listdir(video_path)):
        _, file_extension = os.path.splitext(video_name)
        if file_extension == '':
            continue
        
        path = os.path.join(video_path , video_name)
        video_names.append(path)
    
    if len(video_names) == 0:
        raise FileNotFoundError('동영상 파일이 존재하지 않습니다.')
    return video_names

def main(video_path,output_path, idx=None):
    hands_original_data, pose_original_data, dims = linear_joint(video_path)

    if hands_original_data is not None:

        total_frames = len(hands_original_data)
        left_hand_lost = hands_original_data['left_landmark_x_0'].isna().sum()
        right_hand_lost = hands_original_data['right_landmark_x_0'].isna().sum()
        loss_info_tuple = (total_frames, left_hand_lost, right_hand_lost)
        
        # 프레임 손실률이 25퍼 이상이면 제작 X
        if left_hand_lost > 25 or right_hand_lost > 25:
            return (False, loss_info_tuple)     
        
        check_data = Detect_joint(hands_original_data)
        hand_interpolated_data = spline(check_data.frames)
        pose_interpolated_data = spline_cal(pose_original_data)
        
        hand_df = ( hands_original_data, hand_interpolated_data)
        pose_df = ( pose_original_data, pose_interpolated_data)
        
        if output_path:
            if idx:
                output_path = f'{output_path}{idx+1}'
            create_keypoint_video(output_path, hand_df, pose_df, dims, frame_len=total_frames)
        else:
            visualize_debug(video_path, hand_df, pose_df, dims, loss_info_tuple)
        return (True , loss_info_tuple)
    
    return (False, None)

if __name__ == "__main__":
        
    video_path = None
    output_path = None
    is_log = True
    #f5 로 디버깅 할 때
    if len(sys.argv) == 1:
        # --- 디버그용 ---
        video_path = r'C:\Users\User\Desktop\새 폴더' 
        output_path = 'Test_Debug'
        # -----------------------------
    else:
        
        parser = argparse.ArgumentParser(description="키포인트 추출 기능.")

        parser.add_argument('--video', '-v',
                            type=str, 
                            required=True, 
                            help='처리할 비디오 파일의 경로 또는 폴더경로를 입력')

        parser.add_argument('--output','-o',
                            type=str,
                            required=False,
                            default=None,
                            help='폴더안에 있는 Mp4 파일을 전부 변환 . 예시 Test -> Test.mp4로 출력 예정 \n 값입력 안할 시 영상과 함께 출력됨.')
        
        parser.add_argument('--log','-l',
                            type=bool,
                            required=False,
                            default=True,
                            help='로그 폴더를 작성하지 여부. 기본값 True')

        args = parser.parse_args()
    
        video_path  = args.video 
        output_path = args.output
        is_log = args.log
        
    video_names = check_video(video_path)
    
    if video_names:
        for idx in tqdm(range(len(video_names)) , desc= '진행률'):   
            log_data = main(video_names[idx] , f'{output_path}', idx)
            if is_log:
                write_log(log_data , idx + 1)
    else:
        log_data = main(video_path, output_path)
        
    