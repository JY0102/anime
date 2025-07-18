import os
import numpy as np
import pandas as pd

class Detect_joint:         

    def __init__(self , frames : pd.DataFrame , treshold : float = 8 , bf : int= 5 , is_debug = False , idx = None):       
        
        # 관절 이동거리에 비례해서 이상치 값 추출
        Check_Distance(frames , treshold , bf , is_debug , idx)
        
        # 뼈 길이에 비례해서 이상치 값 추출
        Check_Bone(frames , treshold ,is_debug , idx )
        
              
class Check_Distance:
    
    def __init__(self, frames :pd.DataFrame, treshold: float, bf: int , is_debug = False , idx = None):
        """
        이동 거리에 비례해서 이상치 값을 추출합니다.
        
        Args:
            frames: 좌표 데이터.
            treshold: 이상치 판단을 위한 임계값 배수.
            bf: 이상치 주변을 체크할 프레임 범위.
        
        """
        
        for idx in range(21):
            left_pos  = self.list_extend(frames, idx , 'left')
            right_pos = self.list_extend(frames, idx , 'right')

            self.main_cal(left_pos , right_pos , treshold , bf)

    def main_cal(self, left_pos: list , right_pos: list, treshold: float, bf: int) -> list:
        
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
        left_check_idx  = np.where(left_dis > left_avg * treshold)[0].tolist()
        right_check_idx = np.where(right_dis > right_avg * treshold)[0].tolist()
        
        if len(left_check_idx) or len(right_check_idx):
            print()
                    
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
            
            이동거리 = np.linalg.norm(end - start)
            distances.append(이동거리)
            
        return np.array(distances)
            
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
        position_z = []
        
        position_x.extend(frames[f'{hand_type}_landmark_x_{idx}'].tolist())
        position_y.extend(frames[f'{hand_type}_landmark_y_{idx}'].tolist())
        position_z.extend(frames[f'{hand_type}_landmark_z_{idx}'].tolist())
        
        positions = []                
        for num in range(len(position_x)):
            positions.append(np.array((position_x[num] , position_y[num] , position_z[num])))
            
        return positions     

class Check_Bone:
    
    def __init__(self , frames :pd.DataFrame, treshold: float , is_debug = False , idx = None):
        
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
            left_check_idx  = np.where(left_bones[:, bone_num] < left_avges[bone_num] - 0.02)[0].tolist()
            right_check_idx = np.where(right_bones[:, bone_num] < right_avges[bone_num] - 0.02)[0].tolist()
            
            if len(left_check_idx):
                for num in left_check_idx:
                    left_check[num][bone_num] = True
            if len(right_check_idx):
                for num in right_check_idx:
                    right_check[num][bone_num] = True
        
        x = 5
        left_result = [(idx , sum(left_check[idx])) for idx in range(len(left_check)) if sum(left_check[idx]) >= x]
        right_result = [(idx , sum(right_check[idx])) for idx in range(len(right_check)) if sum(right_check[idx]) >= x]
               
        if is_debug and idx :
            self.write_log(idx , left_result , right_result)
            
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
    
    def write_log(self , idx , left_result ,right_result ):
        
        line = '---------------------------\n'
        with open('debug//left_log.txt', "a", encoding='utf-8') as file:
            import json
            data = json.dumps(left_result , indent=4)
            file.write(f'{idx}\n{data}{line}')
        
        with open('debug//right_log.txt', "a", encoding='utf-8') as file:
            import json
            data = json.dumps(right_result , indent=4)
            file.write(f'{idx}\n{data}{line}')