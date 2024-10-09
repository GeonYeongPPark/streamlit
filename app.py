# import streamlit as st
# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from plot_utils import plot_scene  # 수정된 plot_utils 모듈

# # 경로 설정
# base_path = '/data/datasets/waymo/processing_scneario/processed_scenarios_validation/'

# # 시나리오 파일 목록 가져오기
# pkl_files = [f for f in os.listdir(base_path) if f.endswith('.pkl')]

# # 파일 이름에서 시나리오 ID 추출
# scenario_ids = [f.replace('sample_', '').replace('.pkl', '') for f in pkl_files]

# # 사이드바에 시나리오 ID 선택 위젯 추가
# selected_scenario_id = st.sidebar.selectbox('시나리오 ID를 선택하세요', scenario_ids)

# # max_idx 선택
# max_idx = st.sidebar.slider('예측 경로 개수 (max_idx)', min_value=1, max_value=6, value=6)

# # 선택된 시나리오 ID에 해당하는 파일 이름 생성
# selected_file = f'sample_{selected_scenario_id}.pkl'
# file_path = os.path.join(base_path, selected_file)

# # 프레딕션 파일 경로 설정 (100%와 20% 모두)
# file_path_eda_100 = "/home/geonyeong/park_ws/EDA/output/waymo/eda+100_percent_data/240903_EDA_batch_80_100per_addCyclist_real/eval/eval_with_train/result_adaptive-nms.pkl"
# file_path_mtr_100 = "output/waymo/eda+100_percent_data/240903_EDA_batch_80_100per_baseline/eval/epoch_29/default/result_adaptive-nms.pkl"

# file_path_eda_20 ="output/waymo/eda+20_percent_data/240919_EDA_batch_50_20per_Final/eval/eval_with_train/result_adaptive-nms.pkl"
# file_path_mtr_20 = '/home/geonyeong/park_ws/EDA/output/waymo/eda+20_percent_data/240910_EDA_batch_80_20per_baseline/eval/eval_with_train/result_adaptive-nms.pkl'

# # 파일 존재 여부 확인
# def check_file_exists(file_path):
#     if not os.path.exists(file_path):
#         st.error(f"파일을 찾을 수 없습니다: {file_path}")
#         st.stop()

# check_file_exists(file_path_eda_100)
# check_file_exists(file_path_mtr_100)
# check_file_exists(file_path_eda_20)
# check_file_exists(file_path_mtr_20)
# check_file_exists(file_path)

# # 필요한 데이터 로드 (캐싱 적용)
# @st.cache_data
# def load_data(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#         return data
#     except Exception as e:
#         st.error(f"파일을 로드하는 중 오류 발생: {file_path}\n에러 메시지: {e}")
#         st.stop()

# info = load_data(file_path)
# info_test_eda_100 = load_data(file_path_eda_100)
# info_test_mtr_100 = load_data(file_path_mtr_100)
# info_test_eda_20 = load_data(file_path_eda_20)
# info_test_mtr_20 = load_data(file_path_mtr_20)

# # 시나리오 번호 찾기 함수 정의
# def find_scene_test_num(info_tests, selected_scenario_id):
#     for i in range(len(info_tests)):
#         if info_tests[i][0]['scenario_id'] == selected_scenario_id:
#             return i
#     return None

# # 시나리오 번호 찾기
# scene_test_num_100 = find_scene_test_num(info_test_eda_100, selected_scenario_id)
# scene_test_num_20 = find_scene_test_num(info_test_eda_20, selected_scenario_id)

# if scene_test_num_100 is None or scene_test_num_20 is None:
#     st.error('해당 시나리오 ID를 찾을 수 없습니다.')
# else:
#     # 스타일 설정을 메인 코드에서 한 번만 실행
#     plt.style.use('dark_background')

# # 사이드바에 체크박스 추가
#     plot_all_vehicles = st.sidebar.checkbox('모든 주변 차량 플롯하기', value=False)

#     # 플롯을 위한 Figure와 서브플롯 생성 (2x2)
#     fig, axs = plt.subplots(2, 2, figsize=(20, 20))

#     # 첫 번째 행 (100% 데이터)
#     # EDA-Adaptive (좌측 상단)
#     plot_scene(
#         info,
#         info_test_eda_100,
#         scene_test_num_100,
#         axs[0, 0],
#         "EDA-Adaptive (100%)",
#         max_idx,
#         plot_all_vehicles
#     )

#     # MTR (우측 상단)
#     plot_scene(
#         info,
#         info_test_mtr_100,
#         scene_test_num_100,
#         axs[0, 1],
#         "MTR (100%)",
#         max_idx,
#         plot_all_vehicles
#     )

#     # 두 번째 행 (20% 데이터)
#     # EDA-Adaptive (좌측 하단)
#     plot_scene(
#         info,
#         info_test_eda_20,
#         scene_test_num_20,
#         axs[1, 0],
#         "EDA-Adaptive (20%)",
#         max_idx,
#         plot_all_vehicles
#     )

#     # MTR (우측 하단)
#     plot_scene(
#         info,
#         info_test_mtr_20,
#         scene_test_num_20,
#         axs[1, 1],
#         "MTR (20%)",
#         max_idx,
#         plot_all_vehicles
#     )

#     # 레이아웃 조정
#     plt.tight_layout()

#     # 플롯을 Streamlit에 표시
#     st.pyplot(fig)

#     # 메트릭 계산 함수 정의
#     def minFDE(gt_traj, pred_trajs):
#         indices = {'3s': 40, '5s': 60, '8s': 80}  # 시간 인덱스 (0.1s 간격 가정)
#         min_FDE = {}
#         for key, idx in indices.items():
#             if idx >= len(gt_traj) or idx >= pred_trajs.shape[1]:
#                 continue
#             gt_final = gt_traj[idx, :2]
#             pred_final = pred_trajs[:, idx, :2]
#             displacements = np.linalg.norm(pred_final - gt_final, axis=1)
#             min_FDE[key] = np.min(displacements)
#         return min_FDE

#     def minADE(gt_traj, pred_trajs):
#         indices = {'3s': 40, '5s': 60, '8s': 80}
#         min_ADE = {}
#         for key, idx in indices.items():
#             if idx >= len(gt_traj) or idx > pred_trajs.shape[1]:
#                 continue
#             gt_seg = gt_traj[11:idx+1, :2]
#             pred_seg = pred_trajs[:, :idx-10, :2]
#             errors = np.linalg.norm(pred_seg - gt_seg[None, :, :], axis=2)
#             ade = np.mean(errors, axis=1)
#             min_ADE[key] = np.min(ade)
#         return min_ADE

#     def computeMaxAccelerationJerk(pred_traj):
#         num_modes, num_steps, _ = pred_traj.shape
#         delta_t = 0.5  # 시간 스텝 간격
#         total_duration = (num_steps - 1) * delta_t  # 전체 예측 시간

#         interval_duration = 2.0  # 각 구간의 총 시간은 2초
#         num_intervals = int(total_duration / interval_duration)

#         max_accel_per_mode = np.zeros(num_modes)
#         max_jerk_per_mode = np.zeros(num_modes)

#         for i in range(num_intervals):
#             start_time = i * interval_duration
#             end_time = (i + 1) * interval_duration
#             start_idx = int(start_time / delta_t)
#             end_idx = int(end_time / delta_t) + 1  # 종료 인덱스 포함

#             if end_idx > num_steps:
#                 end_idx = num_steps

#             t_vals = np.linspace(start_time, end_time, end_idx - start_idx)

#             positions = pred_traj[:, start_idx:end_idx, :]  # (num_modes, interval_steps, 2)
#             if positions.shape[1] < 5:
#                 continue  # 데이터 포인트가 충분하지 않으면 건너뜁니다.

#             for j in range(num_modes):
#                 # x와 y에 대해 4차 다항식 피팅
#                 coeffs_x = np.polyfit(t_vals, positions[j, :, 0], 4)
#                 coeffs_y = np.polyfit(t_vals, positions[j, :, 1], 4)

#                 # 다항식의 2차 및 3차 미분 계산
#                 p_x = np.poly1d(coeffs_x)
#                 p_y = np.poly1d(coeffs_y)
#                 accel_x = np.polyder(p_x, 2)(t_vals)
#                 accel_y = np.polyder(p_y, 2)(t_vals)
#                 jerk_x = np.polyder(p_x, 3)(t_vals)
#                 jerk_y = np.polyder(p_y, 3)(t_vals)

#                 # 가속도 및 저크의 크기 계산
#                 accel = np.sqrt(accel_x**2 + accel_y**2)
#                 jerk = np.sqrt(jerk_x**2 + jerk_y**2)

#                 # 각 모드별 최대값 업데이트
#                 max_accel = np.max(accel)
#                 max_jerk = np.max(jerk)

#                 if max_accel > max_accel_per_mode[j]:
#                     max_accel_per_mode[j] = max_accel

#                 if max_jerk > max_jerk_per_mode[j]:
#                     max_jerk_per_mode[j] = max_jerk

#         # 모든 모드에 대한 최대값 계산
#         overall_max_accel = np.max(max_accel_per_mode)
#         overall_max_jerk = np.max(max_jerk_per_mode)

#         return overall_max_accel, overall_max_jerk


#     # 각 모델에 대한 메트릭 계산
   
#     results = []

#     models = [
#         ('EDA-Adaptive (100%)', info_test_eda_100, scene_test_num_100),
#         ('MTR (100%)', info_test_mtr_100, scene_test_num_100),
#         ('EDA-Adaptive (20%)', info_test_eda_20, scene_test_num_20),
#         ('MTR (20%)', info_test_mtr_20, scene_test_num_20)
#     ]

#     for model_name, info_test, scene_test_num in models:
#         agents = info_test[scene_test_num]
#         total_FDE = {'3s': [], '5s': [], '8s': []}
#         total_ADE = {'3s': [], '5s': [], '8s': []}
#         total_acc = []
#         total_jerk = []

#         for agent_info in agents:
#             obj_idx = agent_info['track_index_to_predict']
#             gt_traj = info['track_infos']['trajs'][obj_idx]
#             gt_valid = gt_traj[:, 9] == 1
#             gt_traj = gt_traj[gt_valid]
#             gt_traj = gt_traj[:, :2]  # 위치 정보만 사용

#             pred_trajs = agent_info['pred_trajs']  # (num_modes, num_steps, 7)
#             pred_trajs = pred_trajs[:, :, :2]  # 위치 정보만 사용

#             # 길이 확인
#             if len(gt_traj) < 81 or pred_trajs.shape[1] < 80:
#                 continue  # 데이터가 충분하지 않으면 건너뜁니다.

#             # 메트릭 계산
#             FDE = minFDE(gt_traj, pred_trajs)
#             ADE = minADE(gt_traj, pred_trajs)
#             max_acc, max_jerk = computeMaxAccelerationJerk(pred_trajs)

#             for key in FDE:
#                 total_FDE[key].append(FDE[key])
#             for key in ADE:
#                 total_ADE[key].append(ADE[key])
#             total_acc.append(max_acc)
#             total_jerk.append(max_jerk)

#         # 평균 메트릭 계산
#         avg_FDE = {key: np.mean(total_FDE[key]) if total_FDE[key] else np.nan for key in total_FDE}
#         avg_ADE = {key: np.mean(total_ADE[key]) if total_ADE[key] else np.nan for key in total_ADE}
#         avg_acc = np.mean(total_acc) if total_acc else np.nan
#         avg_jerk = np.mean(total_jerk) if total_jerk else np.nan

#         results.append({
#             'Model': model_name,
#             'FDE': avg_FDE,
#             'ADE': avg_ADE,
#             'Max Acceleration': avg_acc,
#             'Max Jerk': avg_jerk
#         })

#     # 결과를 Streamlit에 표시
#     st.write("## 메트릭 결과")
#     for res in results:
#         st.write(f"**{res['Model']}**")
#         st.write(f"FDE at 3s: {res['FDE'].get('3s', np.nan):.4f}, 5s: {res['FDE'].get('5s', np.nan):.4f}, 8s: {res['FDE'].get('8s', np.nan):.4f}")
#         st.write(f"ADE at 3s: {res['ADE'].get('3s', np.nan):.4f}, 5s: {res['ADE'].get('5s', np.nan):.4f}, 8s: {res['ADE'].get('8s', np.nan):.4f}")
#         st.write(f"Max Acceleration: {res['Max Acceleration']:.4f}")
#         st.write(f"Max Jerk: {res['Max Jerk']:.4f}")
#         st.write("---")
import streamlit as st
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_scene  # 수정된 plot_utils 모듈

# 경로 설정
base_path = '/data/datasets/waymo/processing_scneario/processed_scenarios_validation/'

# 시나리오 파일 목록 가져오기
pkl_files = [f for f in os.listdir(base_path) if f.endswith('.pkl')]

# 파일 이름에서 시나리오 ID 추출
scenario_ids = [f.replace('sample_', '').replace('.pkl', '') for f in pkl_files]

# 사이드바에 시나리오 ID 선택 위젯 추가
selected_scenario_id = st.sidebar.selectbox('시나리오 ID를 선택하세요', scenario_ids)

# max_idx 선택
max_idx = st.sidebar.slider('예측 경로 개수 (max_idx)', min_value=1, max_value=6, value=6)

# 선택된 시나리오 ID에 해당하는 파일 이름 생성
selected_file = f'sample_{selected_scenario_id}.pkl'
file_path = os.path.join(base_path, selected_file)

# 프레딕션 파일 경로 설정 (100%와 20% 모두)
file_path_eda_100 = "/home/geonyeong/park_ws/EDA/output/waymo/eda+100_percent_data/240903_EDA_batch_80_100per_addCyclist_real/eval/eval_with_train/result_adaptive-nms.pkl"
file_path_mtr_100 = "output/waymo/eda+100_percent_data/240903_EDA_batch_80_100per_baseline/eval/epoch_29/default/result_adaptive-nms.pkl"

file_path_eda_20 ="output/waymo/eda+20_percent_data/240919_EDA_batch_50_20per_Final/eval/eval_with_train/result_adaptive-nms.pkl"
file_path_mtr_20 = '/home/geonyeong/park_ws/EDA/output/waymo/eda+20_percent_data/240910_EDA_batch_80_20per_baseline/eval/eval_with_train/result_adaptive-nms.pkl'

# 파일 존재 여부 확인
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        st.stop()

check_file_exists(file_path_eda_100)
check_file_exists(file_path_mtr_100)
check_file_exists(file_path_eda_20)
check_file_exists(file_path_mtr_20)
check_file_exists(file_path)

# 필요한 데이터 로드 (캐싱 적용)
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"파일을 로드하는 중 오류 발생: {file_path}\n에러 메시지: {e}")
        st.stop()

info = load_data(file_path)
info_test_eda_100 = load_data(file_path_eda_100)
info_test_mtr_100 = load_data(file_path_mtr_100)
info_test_eda_20 = load_data(file_path_eda_20)
info_test_mtr_20 = load_data(file_path_mtr_20)

# 시나리오 번호 찾기 함수 정의
def find_scene_test_num(info_tests, selected_scenario_id):
    for i in range(len(info_tests)):
        if info_tests[i][0]['scenario_id'] == selected_scenario_id:
            return i
    return None

# 시나리오 번호 찾기
scene_test_num_100 = find_scene_test_num(info_test_eda_100, selected_scenario_id)
scene_test_num_20 = find_scene_test_num(info_test_eda_20, selected_scenario_id)

if scene_test_num_100 is None or scene_test_num_20 is None:
    st.error('해당 시나리오 ID를 찾을 수 없습니다.')
else:
    # 스타일 설정을 메인 코드에서 한 번만 실행
    plt.style.use('dark_background')

    # 사이드바에 체크박스 추가
    plot_all_vehicles = st.sidebar.checkbox('모든 주변 차량 플롯하기', value=False)

    # 플롯을 위한 Figure와 서브플롯 생성 (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # 첫 번째 행 (100% 데이터)
    # EDA-Adaptive (좌측 상단)
    plot_scene(
        info,
        info_test_eda_100,
        scene_test_num_100,
        axs[0, 0],
        "EDA-Proposed (100%)",
        max_idx,
        plot_all_vehicles
    )

    # MTR (우측 상단)
    plot_scene(
        info,
        info_test_mtr_100,
        scene_test_num_100,
        axs[0, 1],
        "EDA-Baseline (100%)",
        max_idx,
        plot_all_vehicles
    )

    # 두 번째 행 (20% 데이터)
    # EDA-Adaptive (좌측 하단)
    plot_scene(
        info,
        info_test_eda_20,
        scene_test_num_20,
        axs[1, 0],
        "EDA-Proposed (20%)",
        max_idx,
        plot_all_vehicles
    )

    # MTR (우측 하단)
    plot_scene(
        info,
        info_test_mtr_20,
        scene_test_num_20,
        axs[1, 1],
        "EDA-Baseline (20%)",
        max_idx,
        plot_all_vehicles
    )

    # 레이아웃 조정
    plt.tight_layout()

    # 플롯을 Streamlit에 표시
    st.pyplot(fig)

    # 메트릭 계산 함수 정의
    def minFDE(gt_traj, pred_trajs):
        indices = {'3s': 30, '5s': 50, '8s': 79}  # 시간 인덱스 (0.1s 간격 가정)
        min_FDE = {}
        for key, idx in indices.items():
            if idx >= len(gt_traj) or idx >= pred_trajs.shape[1]:
                continue
            gt_final = gt_traj[idx, :2]
            pred_final = pred_trajs[:, idx, :2]
            displacements = np.linalg.norm(pred_final - gt_final, axis=1)
            min_FDE[key] = np.min(displacements)
        return min_FDE

    def minADE(gt_traj, pred_trajs):
        indices = {'3s': 30, '5s': 50, '8s': 79}
        min_ADE = {}
        for key, idx in indices.items():
            if idx >= len(gt_traj) or idx > pred_trajs.shape[1]:
                continue
            gt_seg = gt_traj[11:idx+1, :2]
            pred_seg = pred_trajs[:, :idx-10, :2]
            errors = np.linalg.norm(pred_seg - gt_seg[None, :, :], axis=2)
            ade = np.mean(errors, axis=1)
            min_ADE[key] = np.min(ade)
        return min_ADE

    def computeMaxAccelerationJerk(pred_traj):
        num_modes, num_steps, _ = pred_traj.shape
        delta_t = 0.5  # 시간 스텝 간격
        total_duration = (num_steps - 1) * delta_t  # 전체 예측 시간

        interval_duration = 2.0  # 각 구간의 총 시간은 2초
        num_intervals = int(total_duration / interval_duration)

        max_accel_per_mode = np.zeros(num_modes)
        max_jerk_per_mode = np.zeros(num_modes)

        for i in range(num_intervals):
            start_time = i * interval_duration
            end_time = (i + 1) * interval_duration
            start_idx = int(start_time / delta_t)
            end_idx = int(end_time / delta_t) + 1  # 종료 인덱스 포함

            if end_idx > num_steps:
                end_idx = num_steps

            t_vals = np.linspace(start_time, end_time, end_idx - start_idx)

            positions = pred_traj[:, start_idx:end_idx, :]  # (num_modes, interval_steps, 2)
            if positions.shape[1] < 5:
                continue  # 데이터 포인트가 충분하지 않으면 건너뜁니다.

            for j in range(num_modes):
                # x와 y에 대해 4차 다항식 피팅
                coeffs_x = np.polyfit(t_vals, positions[j, :, 0], 4)
                coeffs_y = np.polyfit(t_vals, positions[j, :, 1], 4)

                # 다항식의 2차 및 3차 미분 계산
                p_x = np.poly1d(coeffs_x)
                p_y = np.poly1d(coeffs_y)
                accel_x = np.polyder(p_x, 2)(t_vals)
                accel_y = np.polyder(p_y, 2)(t_vals)
                jerk_x = np.polyder(p_x, 3)(t_vals)
                jerk_y = np.polyder(p_y, 3)(t_vals)

                # 가속도 및 저크의 크기 계산
                accel = np.sqrt(accel_x**2 + accel_y**2)
                jerk = np.sqrt(jerk_x**2 + jerk_y**2)

                # 각 모드별 최대값 업데이트
                max_accel = np.max(accel)
                max_jerk = np.max(jerk)

                if max_accel > max_accel_per_mode[j]:
                    max_accel_per_mode[j] = max_accel

                if max_jerk > max_jerk_per_mode[j]:
                    max_jerk_per_mode[j] = max_jerk

        # 모든 모드에 대한 최대값 계산
        overall_max_accel = np.max(max_accel_per_mode)
        overall_max_jerk = np.max(max_jerk_per_mode)

        return overall_max_accel, overall_max_jerk

    # 다이버시티 메트릭 함수 추가
    def computeAAE_start_end(pred_traj):
        """
        Average Angle Error (AAE)을 시작점과 끝점에서만 계산합니다.
        각 궤적의 시작 벡터와 끝 벡터 간의 각도 차이를 벡터화 방식으로 계산합니다.
        """
        num_modes, num_steps, _ = pred_traj.shape

        # 시작 벡터 (모든 궤적의 첫 두 점 사이의 벡터)
        start_vecs = pred_traj[:, 1] - pred_traj[:, 0]
        # 끝 벡터 (모든 궤적의 마지막 두 점 사이의 벡터)
        end_vecs = pred_traj[:, -1] - pred_traj[:, -2]

        # 시작 벡터와 끝 벡터의 크기 계산
        start_norms = np.linalg.norm(start_vecs, axis=1)
        end_norms = np.linalg.norm(end_vecs, axis=1)

        # 각 벡터 쌍 사이의 코사인 값을 계산 (시작 벡터와 끝 벡터 각각에 대해)
        start_cosine_matrix = np.dot(start_vecs, start_vecs.T) / (np.outer(start_norms, start_norms) + 1e-8)
        end_cosine_matrix = np.dot(end_vecs, end_vecs.T) / (np.outer(end_norms, end_norms) + 1e-8)

        # 코사인 값을 각도로 변환
        start_angles = np.arccos(np.clip(start_cosine_matrix, -1.0, 1.0))
        end_angles = np.arccos(np.clip(end_cosine_matrix, -1.0, 1.0))

        # 대각선 상의 값(자기 자신과의 비교)을 제외한 각도 값을 사용
        tril_indices = np.tril_indices(num_modes, k=-1)
        start_angle_diffs = start_angles[tril_indices]
        end_angle_diffs = end_angles[tril_indices]

        # 평균 각도 차이를 계산 (시작 벡터와 끝 벡터의 각도 차이 합산 후 평균)
        avg_AAE = np.mean(np.concatenate([start_angle_diffs, end_angle_diffs]))

        return avg_AAE

    def computeAMV(pred_traj):
        num_modes, num_steps, _ = pred_traj.shape
        traj_dis = np.linalg.norm(np.diff(pred_traj, axis=1), axis=2).sum(axis=1)  # (num_modes,)
        discrepancies = []
        for i in range(num_modes):
            for j in range(i+1, num_modes):
                discrepancy = abs(traj_dis[i] - traj_dis[j])
                discrepancies.append(discrepancy)
        if discrepancies:
            AMV = np.mean(discrepancies)
        else:
            AMV = 0
        return AMV

    def computeMinASD(pred_traj, idx_list):
        """
        Minimum Average Self Distance (Min ASD)을 계산합니다.
        특정 시점(idx_list)에서 각 궤적 쌍 간의 평균 거리를 계산하고, 그 중 최소값을 선택합니다.
        """
        num_modes = pred_traj.shape[0]
        minASD_values = []
        for idx in idx_list:
            trajs = pred_traj[:, :idx+1, :]  # (num_modes, idx+1, 2)
            trajs_expanded = trajs[:, np.newaxis, :, :]  # (num_modes, 1, idx+1, 2)
            trajs_diff = trajs_expanded - trajs[np.newaxis, :, :, :]  # (num_modes, num_modes, idx+1, 2)
            distances = np.linalg.norm(trajs_diff, axis=3)  # (num_modes, num_modes, idx+1)
            avg_distances = distances.mean(axis=2)  # (num_modes, num_modes)
            tril_indices = np.tril_indices(num_modes, k=-1)
            pairwise_avg_distances = avg_distances[tril_indices]
            if pairwise_avg_distances.size > 0:
                minASD = np.min(pairwise_avg_distances)
            else:
                minASD = 0
            minASD_values.append(minASD)
        if minASD_values:
            avg_minASD = np.mean(minASD_values)
        else:
            avg_minASD = 0
        return avg_minASD

    def computeMinFSD(pred_traj, idx_list):
        """
        Minimum Final Self Distance (Min FSD)을 계산합니다.
        특정 시점(idx_list)에서 각 궤적 쌍의 최종 위치 간 거리를 계산하고, 그 중 최소값을 선택합니다.
        """
        num_modes = pred_traj.shape[0]
        minFSD_values = []
        for idx in idx_list:
            final_positions = pred_traj[:, idx, :]  # (num_modes, 2)
            diffs = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]  # (num_modes, num_modes, 2)
            distances = np.linalg.norm(diffs, axis=2)  # (num_modes, num_modes)
            tril_indices = np.tril_indices(num_modes, k=-1)
            pairwise_distances = distances[tril_indices]
            if pairwise_distances.size > 0:
                minFSD = np.min(pairwise_distances)
            else:
                minFSD = 0
            minFSD_values.append(minFSD)
        if minFSD_values:
            avg_minFSD = np.mean(minFSD_values)
        else:
            avg_minFSD = 0
        return avg_minFSD

    # 각 모델에 대한 메트릭 계산

    results = []

    models = [
        ('EDA-Proposed (100%)', info_test_eda_100, scene_test_num_100),
        ('EDA-Baseline (100%)', info_test_mtr_100, scene_test_num_100),
        ('EDA-Proposed (20%)', info_test_eda_20, scene_test_num_20),
        ('EDA-Baseline (20%)', info_test_mtr_20, scene_test_num_20)
    ]

    for model_name, info_test, scene_test_num in models:
        agents = info_test[scene_test_num]
        total_FDE = {'3s': [], '5s': [], '8s': []}
        total_ADE = {'3s': [], '5s': [], '8s': []}
        total_acc = []
        total_jerk = []
        total_AAE = []
        total_AMV = []
        total_minASD = []
        total_minFSD = []

        for agent_info in agents:
            obj_idx = agent_info['track_index_to_predict']
            gt_traj = info['track_infos']['trajs'][obj_idx]
            gt_valid = gt_traj[:, 9] == 1
            gt_traj = gt_traj[gt_valid]
            gt_traj = gt_traj[:, :2]  # 위치 정보만 사용

            pred_trajs = agent_info['pred_trajs']  # (num_modes, num_steps, 7)
            pred_trajs = pred_trajs[:, :, :2]  # 위치 정보만 사용

            # 길이 확인
            if len(gt_traj) < 80 or pred_trajs.shape[1] < 80:
                continue  # 데이터가 충분하지 않으면 건너뜁니다.

            # 메트릭 계산
            FDE = minFDE(gt_traj, pred_trajs)
            ADE = minADE(gt_traj, pred_trajs)
            max_acc, max_jerk = computeMaxAccelerationJerk(pred_trajs)

            delta_t = 0.1
            idx_3s = int(3 / delta_t) - 1
            idx_5s = int(5 / delta_t) - 1
            idx_8s = int(8 / delta_t) - 1
            idx_list = [idx_3s, idx_5s, idx_8s]

            max_idx = pred_trajs.shape[1] - 1
            idx_list = [idx for idx in idx_list if idx <= max_idx]

            # 다이버시티 메트릭 계산
            AAE = computeAAE_start_end(pred_trajs)
            AMV = computeAMV(pred_trajs)
            minASD = computeMinASD(pred_trajs, idx_list)
            minFSD = computeMinFSD(pred_trajs, idx_list)

            for key in FDE:
                total_FDE[key].append(FDE[key])
            for key in ADE:
                total_ADE[key].append(ADE[key])
            total_acc.append(max_acc)
            total_jerk.append(max_jerk)
            total_AAE.append(AAE)
            total_AMV.append(AMV)
            total_minASD.append(minASD)
            total_minFSD.append(minFSD)

        # 평균 메트릭 계산
        avg_FDE = {key: np.mean(total_FDE[key]) if total_FDE[key] else np.nan for key in total_FDE}
        avg_ADE = {key: np.mean(total_ADE[key]) if total_ADE[key] else np.nan for key in total_ADE}
        avg_acc = np.mean(total_acc) if total_acc else np.nan
        avg_jerk = np.mean(total_jerk) if total_jerk else np.nan
        avg_AAE = np.mean(total_AAE) if total_AAE else np.nan
        avg_AMV = np.mean(total_AMV) if total_AMV else np.nan
        avg_minASD = np.mean(total_minASD) if total_minASD else np.nan
        avg_minFSD = np.mean(total_minFSD) if total_minFSD else np.nan

        results.append({
            'Model': model_name,
            'FDE': avg_FDE,
            'ADE': avg_ADE,
            'Max Acceleration': avg_acc,
            'Max Jerk': avg_jerk,
            'AAE': avg_AAE,
            'AMV': avg_AMV,
            'Min ASD': avg_minASD,
            'Min FSD': avg_minFSD
        })

    # 결과를 Streamlit에 표시
    st.write("## 메트릭 결과")
    st.write(f'{"sample_"+selected_scenario_id}')
    for res in results:
        st.write("### Model Evaluation Summary")
        st.write(f"**Model: {res['Model']}**")

        # Accuracy Section
        st.write("#### Accuracy Metrics")
        st.write(f"- **FDE** at 3s: {res['FDE'].get('3s', np.nan):.4f}, 5s: {res['FDE'].get('5s', np.nan):.4f}, 8s: {res['FDE'].get('8s', np.nan):.4f}")
        st.write(f"- **ADE** at 3s: {res['ADE'].get('3s', np.nan):.4f}, 5s: {res['ADE'].get('5s', np.nan):.4f}, 8s: {res['ADE'].get('8s', np.nan):.4f}")

        # Dynamics Section
        st.write("#### Dynamics Metrics")
        st.write(f"- **Max Acceleration**: {res['Max Acceleration']:.4f} m/s²")
        st.write(f"- **Max Jerk**: {res['Max Jerk']:.4f} m/s³")

        # Diversity Section
        st.write("#### Diversity Metrics")
        st.write(f"- **AAE**: {res['AAE']*100:.4f}%")
        st.write(f"- **AMV**: {res['AMV']:.4f}")
        st.write(f"- **Min ASD**: {res['Min ASD']:.4f}")
        st.write(f"- **Min FSD**: {res['Min FSD']:.4f}")

        st.write("---")

