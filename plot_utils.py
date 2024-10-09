import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_polyline(ax, polyline, color, linewidth, linestyle='-'):
    ax.plot(polyline[:, 1], polyline[:, 0], color=color, linewidth=linewidth, linestyle=linestyle)

def plot_polygon(ax, polygon, edgecolor, facecolor, alpha=0.1):
    polygon_patch = Polygon(polygon, closed=True, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
    ax.add_patch(polygon_patch)

def global2local(global_coords, ref_x, ref_y, ref_heading):
    coords = global_coords - np.array([ref_x, ref_y])
    rotation_matrix = np.array([
        [np.cos(ref_heading), -np.sin(ref_heading)],
        [np.sin(ref_heading),  np.cos(ref_heading)]
    ])
    local_coords = coords @ rotation_matrix
    return local_coords

def plot_scene(info, info_test, scene_test_num, ax, title_suffix, max_idx_plot=6, plot_all_vehicles=False):
    scene_id = info['scenario_id']
    track_infos = info['track_infos']
    track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])

    map_info = info['map_infos']

    sdc_track_index = info['sdc_track_index']
    current_time_index = info['current_time_index']
    sdc_obj_traj = track_infos['trajs'][sdc_track_index]

    sdc_curr_x = sdc_obj_traj[current_time_index, 0]
    sdc_curr_y = sdc_obj_traj[current_time_index, 1]
    sdc_curr_heading = sdc_obj_traj[current_time_index, 6]
    
    # ax의 배경색 설정
    ax.set_facecolor('#333333')

    # 제목 색상을 흰색으로 설정
    ax.set_title(f'{scene_id} ({title_suffix})', color='white')

    if 'lane' in map_info:
        for lane in map_info['lane']:
            polyline_index = lane['polyline_index']
            centerline_polyline = global2local(
                map_info['all_polylines'][polyline_index[0]:polyline_index[1], :2],
                sdc_curr_x, sdc_curr_y, sdc_curr_heading
            )
            plot_polyline(ax, centerline_polyline, 'white', 0.5, '--')

    if 'crosswalk' in map_info:
        for crosswalk_info in map_info['crosswalk']:
            polyline_index = crosswalk_info['polyline_index']
            crosswalk_polyline = global2local(
                map_info['all_polylines'][polyline_index[0]:polyline_index[1], :2],
                sdc_curr_x, sdc_curr_y, sdc_curr_heading
            )
            crosswalk_polyline[:, [0, 1]] = crosswalk_polyline[:, [1, 0]]
            plot_polygon(ax, crosswalk_polyline, 'yellow', 'yellow')

    if plot_all_vehicles:
        obj_indices = range(len(track_infos['object_id']))  # 모든 차량
    else:
        obj_indices = np.append(sdc_track_index, track_index_to_predict)  # 예측 대상 차량 및 SDC

    # 차량 플롯
    for obj_idx in obj_indices:
        obj_type = track_infos['object_type'][obj_idx]
        obj_traj = track_infos['trajs'][obj_idx]
        valid_indices = obj_traj[:, 9] == 1
        obj_traj = obj_traj[valid_indices]
        times = np.arange(len(obj_traj))

        if len(obj_traj) == 0:
            continue  # 유효한 트랙이 없으면 건너뜁니다.

        if obj_type == 'TYPE_VEHICLE':
            if obj_idx not in obj_indices:
                cmap = 'YlOrRd'
                edgecolor = 'orange'
            else:
                cmap = 'YlOrRd'
                edgecolor = 'red' if obj_idx != sdc_track_index else 'deepskyblue'
            L, W = np.mean(obj_traj[:, 3]), np.mean(obj_traj[:, 4])
        elif obj_type == 'TYPE_PEDESTRIAN':
            cmap = 'Blues'
            edgecolor = 'blue'
            L, W = np.mean(obj_traj[:, 3]), np.mean(obj_traj[:, 4])
        elif obj_type == 'TYPE_CYCLIST':
            cmap = 'YlGn'
            edgecolor = 'green'
            L, W = np.mean(obj_traj[:, 3]), np.mean(obj_traj[:, 4])
        else:
            continue  # 지원되지 않는 객체 유형은 건너뜁니다.

        points = global2local(obj_traj[:, :2], sdc_curr_x, sdc_curr_y, sdc_curr_heading)
        color_values = times

        if len(points) <= current_time_index:
            continue

        ax.scatter(points[:, 1], points[:, 0], c=color_values[:], cmap=cmap, marker='o', s=1)
        ax.scatter(points[-1, 1], points[-1, 0], c=color_values[-1], cmap=cmap, marker='*', s=30, zorder = 10)
        x_end, y_end = points[-1, 1], points[-1, 0]
        heading = obj_traj[-1, 6] - sdc_curr_heading

        curr_x, curr_y = points[current_time_index, 0], points[current_time_index, 1]
        heading = obj_traj[current_time_index, 6]
        rot_mat = np.array([
            [np.cos(heading - sdc_curr_heading), -np.sin(heading - sdc_curr_heading)],
            [np.sin(heading - sdc_curr_heading),  np.cos(heading - sdc_curr_heading)]
        ])

        x_rec = [-L / 2, L / 2, L / 2, -L / 2]
        y_rec = [-W / 2, -W / 2, W / 2, W / 2]
        veh = np.dot(rot_mat, [x_rec, y_rec]) + np.array([[curr_x], [curr_y]])

        # 텍스트 레이블 조건부 표시
        if obj_idx in track_index_to_predict or obj_idx == sdc_track_index:
            ax.text(curr_y, curr_x + W / 2 + 0.5, f'{obj_idx}', color='white', ha='center', va='center')

        rect = Polygon(
            ((veh[1, 0], veh[0, 0]), (veh[1, 1], veh[0, 1]), (veh[1, 2], veh[0, 2]), (veh[1, 3], veh[0, 3])),
            closed=True, edgecolor=edgecolor, facecolor='None',
            label=f'{obj_idx}' if obj_idx in track_index_to_predict else None
        )
        ax.add_patch(rect)

        x_tri = [0, L / 2, L / 2, 0]
        y_tri = [-W/2, 0, 0, W/2]
        veh_head = np.dot(rot_mat, [x_tri, y_tri]) + np.array([[curr_x], [curr_y]])
        tri = Polygon(
            ((veh_head[1, 0], veh_head[0, 0]), (veh_head[1, 1], veh_head[0, 1]),
             (veh_head[1, 2], veh_head[0, 2]), (veh_head[1, 3], veh_head[0, 3])),
            closed=True, edgecolor=edgecolor, facecolor='None'
        )
        ax.add_patch(tri)

    # 예측 경로 그리기
    color_list = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'pink']

    for i in range(len(info_test[scene_test_num])):
        pred_trajectory = info_test[scene_test_num][i]['pred_trajs']
        num_trajs = min(max_idx_plot, len(pred_trajectory))
        for idx in range(num_trajs):
            points = global2local(
                pred_trajectory[idx][:, :2], sdc_curr_x, sdc_curr_y, sdc_curr_heading
            )
            ax.scatter(
                points[:, 1],
                points[:, 0],
                alpha=0.1,
                marker='o',
                s=2,
                color=color_list[i % len(color_list)],
            )
            

    limy = (-100, 100)
    limx = (-100, 100)

    ax.set_xlim(limx)
    ax.invert_xaxis()
    ax.set_ylim(limy)

    # 범례 중복 제거
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # 축 레이블과 눈금 색상 설정
    ax.set_xlabel('Lateral', color='white')
    ax.set_ylabel('Longitudinal', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
