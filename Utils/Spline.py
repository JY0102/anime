import pandas as pd

from scipy.interpolate import CubicSpline

def interpolate_with_cubicspline(df):
    """
    DataFrame의 각 열에 대해 CubicSpline 보간을 적용하는 함수.
    """
    interpolated_df = df.copy()
    
    # 각 좌표계 컬럼에 대해 반복
    for col in interpolated_df.columns:
        series = interpolated_df[col]
        
        # 유효한 데이터(NaN이 아닌 값)와 해당 프레임(인덱스) 찾기
        valid_indices = series.dropna().index
        valid_values = series.loc[valid_indices].values
        
        # 보간할 데이터가 있는지, 그리고 Cubic Spline을 만들기에 충분한 데이터가 있는지 확인
        if series.isna().any() and len(valid_indices) > 3:
            # CubicSpline 함수 생성 (bc_type='natural'은 양 끝 경계 조건을 자연스럽게 처리)
            spline = CubicSpline(valid_indices, valid_values, bc_type='natural')
            
            # 비어있는(NaN) 프레임 찾기
            missing_indices = series.isna().index
            
            # 비어있는 프레임의 값을 스플라인 함수로 추정
            interpolated_values = spline(missing_indices)
            
            # 추정된 값으로 비어있는 데이터 채우기
            interpolated_df.loc[missing_indices, col] = interpolated_values
            
    # CubicSpline은 영상 시작/끝의 NaN은 처리 못하므로, ffill/bfill로 한 번 더 채워줌
    interpolated_df.ffill(inplace=True)
    interpolated_df.bfill(inplace=True)

    return interpolated_df

    
