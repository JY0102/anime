import os
import json
import numpy as np
import pandas as pd
from typing import Literal


class Outlier :
    
    idx = None
    hand_type = None
    rotation = None
    before_rotation = None
    after_rotation = None
    # 위험도 레벨
    level = 0

    def __init__(self,idx , hand_type : Literal ['left','right'], rotation, before_rotation, after_rotation):
        
        """ 이상치 값 측정용 객체
        
        매개변수
        --------------        
        hand_type : 왼손 , 오른손 ( left, right)
        rotation : 문제가 있는 인덱스의 회전 값.
        before : 이전 프레임의 회전값
        after : 이후 프레임의 회전값
        """
        
        self.idx = idx
        self.hand_type = hand_type
        self.rotation = rotation
        self.before_rotation = before_rotation
        self.after_rotation = after_rotation

        self.cal_level()
        self.output = self.out_string()
        
    def cal_level(self):
        
        h_type = self.hand_type
        
        rt = self.rotation
        
        left_b_rt, right_b_rt = self.before_rotation
        left_a_rt, right_a_rt = self.after_rotation
        
        if h_type == 'left':      
            if abs(right_b_rt - rt) < 30 :
                self.level += 1
            elif abs(abs(right_b_rt) - abs(rt)) < 15:
                self.level += 1
            
            if abs(right_a_rt - rt) < 30 :
                self.level += 1       
            elif abs(abs(right_a_rt) - abs(rt)) < 15:
                self.level += 1         
        elif h_type == 'right':            
            if abs(left_b_rt - rt) < 30 :
                self.level += 1
            elif abs(abs(left_b_rt) - abs(rt)) < 15:
                self.level += 1
            
            if abs(left_a_rt - rt) < 30 :
                self.level += 1
            elif abs(abs(left_a_rt) - abs(rt)) < 15:
                self.level += 1 
    
    def out_string(self):
        프레임 = self.idx
        감지된손 = f'{self.hand_type}'
        회전값 = f'{self.rotation}'
        이전_회전값 = f'{self.before_rotation[0]}  |  {self.before_rotation[1]}'
        이후_회전값 = f'{self.after_rotation[0]}  |  {self.after_rotation[1]}'
        위험도 = f'{self.level}'
        
        result = {
            '프레임' : 프레임,
            '감지된손' : 감지된손,
            '회전값' : 회전값,
            '이전 회전값' : 이전_회전값,
            '이후 회전값' : 이후_회전값,
            '위험도' : 위험도,
        }
        
        return json.dumps(result, indent= 4, ensure_ascii=False)
              
class Detect_joint:         

    frames = None
    def __init__(self , df : pd.DataFrame, treshold : float = 8, is_debug = False, debug_data = None):       
        """MediaPipe에서 추출한 RawData를 알고리즘을 이용하여 일부 보간

        Args:
            df (pd.DataFrame): MediaPipe에서 출력한 데이터프레임
            treshold (float, optional): 임계값.해당 값이 크면 클 수록 이상치 값 감지가 둔해짐.권장 값 7 ~ 9
            is_debug (bool, optional): 디버깅용인지 여부. True라면 Log 작성
            debug_data (_type_, optional): 디버깅에 필요한 데이터
        """
        self.frames = df
        # 관절 이동거리에 비례해서 이상치 값 추출
        self.distance = Interpolation_Distance(df , treshold , is_debug , debug_data)
        
        # 뼈 길이에 비례해서 이상치 값 추출
        # Interpolation_Bone(df , treshold ,is_debug , debug_data)
        
        self.rotation = Interpolation_Rotation(df , treshold , is_debug , debug_data)
        
        self.delete_data()
        
        self.change_data()
        
    def delete_data(self):
        for idx in self.distance.left_last:
            delete_column = self.frames.loc[idx].filter(like='left',axis=0).index 
            self.frames.loc[idx, delete_column] = np.nan                 

    def change_data(self):
        """왼손 오른손이 뒤바꼈을 때 다시 제자리로 돌려 놓음.
        """
        
        pair_hand = {
            'left': 'right',
            'right': 'left'
        }
        
        for frame,hand_type in self.rotation.warning_index_list:
            for idx in range(21):

                self.frames.loc[frame, f'{hand_type}_landmark_x_{idx}'], self.frames.loc[frame, f'{pair_hand[hand_type]}_landmark_x_{idx}'] = \
                self.frames.loc[frame, f'{hand_type}_landmark_x_{idx}'], self.frames.loc[frame, f'{pair_hand[hand_type]}_landmark_x_{idx}']
                
                
                self.frames.loc[frame, f'{hand_type}_landmark_y_{idx}'], self.frames.loc[frame, f'{pair_hand[hand_type]}_landmark_y_{idx}'] = \
                self.frames.loc[frame, f'{hand_type}_landmark_y_{idx}'], self.frames.loc[frame, f'{pair_hand[hand_type]}_landmark_y_{idx}']

                self.frames.loc[frame, f'{hand_type}_landmark_x_{idx}'] = np.NAN
                self.frames.loc[frame, f'{hand_type}_landmark_y_{idx}'] = np.NAN
                
class Interpolation_Distance:
    """이동 거리에 비례해서 이상치 값을 추출합니다.
    """
    
    left_last = None
    right_last = None
    
    def __init__(self, frames :pd.DataFrame, treshold: float, is_debug = False , debug_data = None):
        """        
        Args:
            frames: 좌표 데이터.
            treshold: 이상치 판단을 위한 임계값 배수.
            bf: 이상치 주변을 체크할 프레임 범위.
        
        """
        
        가로 = 21
        세로 = len(frames)
        left_check = [[False for _ in range(가로)] for _ in range(세로)]
        right_check = [[False for _ in range(가로)] for _ in range(세로)]
        
        for bone_num in range(21):
            left_pos  = self.list_extend(frames, bone_num , 'left')
            right_pos = self.list_extend(frames, bone_num , 'right')

            left_check_idx , right_check_idx = self.main_cal(left_pos , right_pos , treshold)
            
            if len(left_check_idx):
                for num in left_check_idx:
                    left_check[num][bone_num] = True
            if len(right_check_idx):
                for num in right_check_idx:
                    right_check[num][bone_num] = True
                    
        left_result = [(num , sum(left_check[num])) for num in range(len(left_check)) if sum(left_check[num]) >= treshold + 5]
        right_result = [(num , sum(right_check[num])) for num in range(len(right_check)) if sum(right_check[num]) >= treshold + 5]

        self.left_last = self.chose_idx(left_result)
        self.right_last = self.chose_idx(right_result)
        
        if is_debug and debug_data :
            self.write_log(debug_data[0] , self.left_last , self.right_last , debug_data[1] , debug_data[2])

    def main_cal(self, left_pos: list , right_pos: list, treshold: float ) -> list:
        
        """
        주 거리 계산 알고리즘 , 평균거리 보다 임계값보다 큰 값의 인덱스 번호 저장
        
        Args:
            joints: 좌표 데이터 리스트.
            treshold: 이상치 판단을 위한 임계값 배수.
            bf: 이상치 주변을 체크할 프레임 범위.
        Returns:
            이상치로 판단된 프레임의 인덱스 리스트.
        """
        
        # 이동거리 값
        left_dis  = self.cal_distance(left_pos)
        right_dis = self.cal_distance(right_pos)
        
        # 평균 이동거리
        left_avg  = np.nanmean(left_dis)
        right_avg = np.nanmean(right_dis)
        
        # 평균 이동거리 * 임계값 보다 큰 이동거리 인덱스 번호 추출
        left_check_idx  = np.where(left_dis > left_avg * 2)[0].tolist()
        right_check_idx = np.where(right_dis > right_avg * 2)[0].tolist()
        
        return left_check_idx , right_check_idx
                    
    def cal_distance(self , joints : list) -> list:
        """ 
        이동거리 계산

        Args:
            joints (list): 좌표 정보 [( x , y, z )]

        Returns:
            list: 이동거리
        """
        distances = []
        
        for idx in range(len(joints) - 1):
            start = joints[idx]
            end   = joints[idx+1]
            
            if np.isnan(start).any() or np.isnan(end).any():
                distances.append(np.nan)
                continue
            
            이동거리 = np.linalg.norm(end - start)
            distances.append(이동거리)
            
        return np.array(distances)
   
    def chose_idx(self , last):
        
        def check():
            if len(group) > 1:
                group.clear()
            elif len(group) != 0:
                result.append(group[0] + 1)
                group.clear()
                    
        group = []
        result = []
            
        for i in range(0 , len(last) -1):
            if last[i+1][0] - last[i][0] == 1:
                group.append(last[i][0])
            else :
                check()
        if len(last) == 2:
            check()
        
        return result
                        
    def list_extend(self, frames : pd.DataFrame , idx : int , hand_type : str) -> list:        
        """
        MediaPipe에서 뽑은 좌표 정규화
        
        Args:
            idx: 뼈위치 인덱스값
            hand_type: 왼손 , 오른손 ( left , right)
            
        Returns:
            list : 특정관절의 프레임별 좌표 값 
            list[N] -> N프레임일 때의 특정 관절의 좌표 값
            [np.array[x , y, z]]            
        """
        
        position_x = []
        position_y = []
        
        position_x.extend(frames[f'{hand_type}_landmark_x_{idx}'].tolist())
        position_y.extend(frames[f'{hand_type}_landmark_y_{idx}'].tolist())
        
        positions = []                
        for num in range(len(position_x)):
            positions.append(np.array((position_x[num] , position_y[num])))

        return positions     
     
    def write_log(self , idx , left_result ,right_result , left_lost , right_lost ):
        base_dir = 'debug//distance//'
        path  = os.path.join(base_dir , 'log.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        if len(left_result) > 0 or len(right_result) > 0:
            with open(path, "a", encoding='utf-8') as file:
                
                file.write(f'{idx} : 프레임손실율(왼손/오른손) : {left_lost}/{right_lost}\n')
                if len(left_result) > 0:
                    left_data  = json.dumps(left_result)                    
                    file.write(f'왼손: {left_data}\n')
                if len(right_result) > 0:
                    right_data = json.dumps(right_result)
                    file.write(f'오른손: {right_data}\n')
                file.write(f'\n\n')

class Interpolation_Bone:
    """
    뼈 길이 기반으로 이상치 데이터 추출 ( 개발 중)
    """
    def __init__(self , frames :pd.DataFrame, treshold: float , is_debug = False , debug_data = None):
        
        """
        평균 뼈 길이를 계산 후 평균 뼈 길이보다 작을 시 이상치 값 판단.
        """
        
        self.bone_mapping = [ 
            (0,1) , (1,2) , (2,3) , (3,4),
            (0,5) , (5,6) , (6,7) , (7,8),
            (0,9) , (9,10) , (10,11) , (11,12),
            (0,13) , (13,14) , (14,15) , (15,16),
            (0,17) , (17,18) , (18,19) , (19,20)
        ]
            
        
        left_frames  = self.list_extend(frames, 'left')
        right_frames = self.list_extend(frames, 'right')
            
        left_bones  = self.cal_bones(left_frames)
        right_bones = self.cal_bones(right_frames)
        
        left_avges  = self.avg_bones(left_bones)
        right_avges = self.avg_bones(right_bones)

        가로 = 20
        세로 = len(frames)
        
        left_check = [[False for _ in range(가로)] for _ in range(세로)]
        right_check = [[False for _ in range(가로)] for _ in range(세로)]
        
        for bone_num in range(20):
            left_check_idx  = np.where(left_bones[:, bone_num] < left_avges[bone_num] - 0.015)[0].tolist()
            right_check_idx = np.where(right_bones[:, bone_num] < right_avges[bone_num] - 0.015)[0].tolist()
            
            if len(left_check_idx):
                for num in left_check_idx:
                    left_check[num][bone_num] = True
            if len(right_check_idx):
                for num in right_check_idx:
                    right_check[num][bone_num] = True
        
        left_result = [(num , sum(left_check[num])) for num in range(len(left_check)) if sum(left_check[num]) >= treshold]
        right_result = [(num , sum(right_check[num])) for num in range(len(right_check)) if sum(right_check[num]) >= treshold]
              
        if is_debug and debug_data :
            self.write_log(debug_data[0] , left_result , right_result , debug_data[1] , debug_data[2])
            
    def cal_bones(self , frames : list) -> np.array:
        """ 
        뼈길이 계산

        Args:
            frames (list): 좌표 정보 [ [( x , y, z )] , , , ]

        Returns:
            np.array: N프레임의 모든 뼈 길이
        """
        
        bones = []
        
        for frame in frames:
            bone = []
            for start_idx , end_idx in self.bone_mapping:
                start = frame[start_idx]
                end = frame[end_idx]
                
                뼈길이 = np.linalg.norm(end - start)
                bone.append(뼈길이)
            
            bones.append(bone)
        return np.array(bones)
    
    def avg_bones(self , bones : np.array) -> list:
        """
        특정 뼈의 평균길이 값.

        Returns:
            list : N번째 인덱스의 평균 뼈 길이
        """
        average = []
        for idx in range(20):
            sel_bones = bones[:, idx]
            average.append(np.nanmean(sel_bones))
            
        return average
                
    def list_extend(self, frames : pd.DataFrame ,hand_type : str) -> list:        
        """
        MediaPipe에서 뽑은 좌표 정규화
        
        Args:
            idx: 뼈위치 인덱스값
            hand_type: 왼손 , 오른손 ( left , right)
            
        Returns:
            list : 특정관절의 프레임별 좌표 값 
            list[N] -> N프레임일 때의 모든 관절의 좌표 값
            [ [np.array[x , y, z]] . . . ]      
        """
        result = []
        joint_data = frames.filter(like=f'{hand_type}', axis=1)

        for idx in frames.index:
                
            joint = joint_data.loc[idx]
            
            pos = []
            for idx in range(21):
                pos_x = joint[f'{hand_type}_landmark_x_{idx}']
                pos_y = joint[f'{hand_type}_landmark_y_{idx}']
                # pos_z = joint[f'{hand_type}_landmark_z_{idx}']
                
                # joints.append(np.array((pos_x , pos_y , pos_z)))
                pos.append(np.array((pos_x , pos_y )))
            result.append(pos)                   
        
        return result     

    def write_log(self , idx , left_result ,right_result , left_lost , right_lost ):
 
        base_dir = 'debug//bone//'
        path  = os.path.join(base_dir , 'log.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        if len(left_result) > 0 or len(right_result) > 0:
            with open(path, "a", encoding='utf-8') as file:
                
                file.write(f'{idx}프레임손실율(왼손/오른손) : {left_lost}/{right_lost}\n')
                if len(left_result) > 0:
                    left_data  = json.dumps(left_result)                    
                    file.write(f'왼손: {left_data}\n')
                if len(right_result) > 0:
                    right_data = json.dumps(right_result)
                    file.write(f'오른손: {right_data}\n')
                file.write(f'\n\n')

class Interpolation_Rotation:
    
    warning_index_list = []
    
    def __init__(self , frames :pd.DataFrame, treshold: float , is_debug = False , debug_data = None):
        
        """
        손목과 엄지의 방향을 계산하여 , 왼손과 오른손의 값이 뒤바뀌었을 때 방향벡터값을 기준으로 원래 자리를 다시 되찾아가는 알고리즘.
        """
        
        길이 = len(frames)
        
        left_frames  = self.list_extend(frames, 'left')
        right_frames = self.list_extend(frames, 'right')
        
        level_frames = [None for _ in range(길이)]
        warning_list = [None for _ in range(길이)]
        
        for check_idx in range(길이):
            
            left_unavailable  = np.isnan(left_frames[check_idx]).all()
            right_unavailable = np.isnan(right_frames[check_idx]).all()
            
            # 양손 중 한손만 존재한다면
            if left_unavailable ^ right_unavailable:
                check_info = ()
                
                if not left_unavailable:
                    rotation = self.cal_rotation(left_frames[check_idx])
                    check_info = ('left', rotation)
                elif not right_unavailable:
                    rotation = self.cal_rotation(right_frames[check_idx])
                    check_info = ('right', rotation)
                
                before_rotation = ( self.cal_rotation(left_frames[check_idx -1]) , self.cal_rotation(right_frames[check_idx -1]) )
                after_rotation  = ( self.cal_rotation(left_frames[check_idx +1]) , self.cal_rotation(right_frames[check_idx +1]) )
                
                out = Outlier(check_idx, *check_info, before_rotation, after_rotation)
                level_frames[check_idx] = out.level
                warning_list[check_idx] = out
             
        self.warning_index_list = self.guess_warning(level_frames, warning_list)
        if is_debug :      
            # self.write_log_test(debug_data[0], warning_index_list, warning_list)
            self.write_log(debug_data[0], self.warning_index_list)
        
    def cal_rotation(self, keypoints):
        
        if not keypoints:
            return None
            
        손목 = keypoints[0]
        엄지 = keypoints[1]
        
        vector = 엄지 - 손목

        angle_rad = np.arctan2(vector[1], vector[0])
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def guess_warning(self, level_frames, warning_list: list[Outlier]) -> list:
        """ 이상치 값 레벨로 판단해서 고위험군 이상치 값 판별
        
        Return
        -------------
        고위험군 프레임 리스트값
        """

        start_idx = None
        warning_index_list = []
        check_idx = 0
        
        for idx in range(len(level_frames)):
            if level_frames[idx] is None:
                start_idx = None
                check_idx = 0
            else:
                match level_frames[idx]:
                    case 0:
                       if start_idx:
                           check_idx += 1
                    case 1:
                        if start_idx:
                            check_idx += 1
                            for i in range(check_idx):
                                warning_index_list.append((idx - i, warning_list[idx - i].hand_type))
                            start_idx == None
                            check_idx = 0
                        else:
                            start_idx = idx
                            check_idx += 1
                    case 2:
                        level_frames[idx+1] = None
                        warning_index_list.append((idx, warning_list[idx].hand_type))
                
        warning_index_list.sort(key=lambda x: x[0])
        return warning_index_list        
        
    def list_extend(self , frames : pd.DataFrame , hand_type : str) -> list:        
        """
        MediaPipe에서 뽑은 좌표 정규화
        
        Args:
            idx: 뼈위치 인덱스값
            hand_type: 왼손 , 오른손 ( left , right)
            
        Returns:
            list : 프레임별 모든 관절 좌표 값 
            list[N] -> N프레임일 때의 모든 관절의 좌표 값
            [ [np.array[x , y, z]] . . . ]      
        """
        result = []
        joint_data = frames.filter(like=f'{hand_type}', axis=1)

        for idx in frames.index:
                
            joint = joint_data.loc[idx]
            
            pos = []
            for idx in range(21):
                pos_x = joint[f'{hand_type}_landmark_x_{idx}']
                pos_y = joint[f'{hand_type}_landmark_y_{idx}']
                # pos_z = joint[f'{hand_type}_landmark_z_{idx}']
                
                # joints.append(np.array((pos_x , pos_y , pos_z)))
                pos.append(np.array((pos_x , pos_y )))
            result.append(pos)                   
        
        return result     

    def write_log(self, idx, index_list):
        base_dir = 'debug//rotation//'
        path  = os.path.join(base_dir , 'log.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        
        with open(path, "a", encoding='utf-8') as file:
            
            file.write(f'{idx} 번째 영상\n')
            file.write(f'문제된 프레임번호 {json.dumps(index_list)}\n\n')

    def write_log_test(self,idx,  index_list : list , warning_list : list[Outlier]):
        """조금 더 자세하게 로그를 작성
        """
        base_dir = 'debug//rotation//'
        path  = os.path.join(base_dir , 'log_test.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        with open(path, "a", encoding='utf-8') as file:
            
            file.write(f'{json.dumps(index_list)}\n\n')
            
            for obj in warning_list:
                file.write(f'{obj.output}\n')
                
            file.write(f'\n\n')
  
                