import numpy as np
import pandas as pd

class Detect_joint:         

    def __init__(self , frames = pd.DataFrame , treshold = 8 , bf = 5):
        for idx in range(21):
            left_pos = self.list_extend(frames, idx , 'left')
            right_pos = self.list_extend(frames, idx , 'right')

            # 이동거리에 비례해서 이상치 값 추출
            Check_Distance(left_pos , right_pos , treshold , bf)
            
            # 뼈 길이에 비례해서 이상치 값 추출
        
    def list_extend(self, frames , idx , hand_type):
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
              
class Check_Distance:
    
    def __init__(self , left_pos , right_pos , treshold , bf):
        """
        이동거리에 비례해서 이상치 값 추출
        
        매개변수:
        
            joints | 좌표 데이터 ( List )
            treshold | 평균 이동 거리와 어느정도 차이가 나야 이상치 데이터로 특정할지
            bf  | 이상치 데이터가 검출 됐을 때 주변 몇 프레임까지 체크할지.
            
        """

        # 이동거리 값
        left_dis = self.cal_distance(left_pos)
        right_dis = self.cal_distance(right_pos)
        
        # 평균 이동거리
        left_avg = np.nanmean(left_dis)
        right_avg = np.nanmean(right_dis)
        
        # 평균 이동거리 * 임계값 보다 큰 이동거리 인덱스 번호 추출
        left_check_idx = np.where(left_dis > left_avg * treshold)[0].tolist()
        right_check_idx = np.where(right_dis > right_avg * treshold)[0].tolist()
        
        if len(left_check_idx) or len(right_check_idx):
            print('h')
    
                    
    def cal_distance(self , joints):
        distances = []
        
        for idx in range(len(joints) - 1):
            start = joints[idx]
            end = joints[idx+1]
            
            이동거리 = np.linalg.norm(end - start)
            distances.append(이동거리)
            
        return np.array(distances)
  
        
        
        
        