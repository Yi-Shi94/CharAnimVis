import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    cnt = 0
    myStack = []
    root_joint_name = None
    with open(bvh_file_path, 'r') as file_obj:
        for line in file_obj:
            lineList = line.split()
            if (lineList[0] == "{"):
                myStack.append(cnt)
                cnt += 1

            if (lineList[0] == "}"):
                myStack.pop()

            if (lineList[0] == "OFFSET"):
                joint_offset.append([float(lineList[1]), float(lineList[2]), float(lineList[3])])

            if (lineList[0] == "JOINT"):
                joint_name.append(lineList[1])
                joint_parent.append(myStack[-1])
            
            elif (lineList[0] == "ROOT"):
                joint_name.append(lineList[1])
                joint_parent.append(-1)
                root_joint_name = lineList[1]

            elif (lineList[0] == "End"):
                joint_name.append(joint_name[-1] + '_end')
                joint_parent.append(myStack[-1])

    joint_offset = np.array(joint_offset).reshape(-1, 3)
    root_offset = np.array(joint_offset[joint_name.index(root_joint_name)])
    joint_offset[joint_name.index(root_joint_name)] *= 0
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    channals_num = (len(motion_data[frame_id]) -3) // 3
    root_position = motion_data[frame_id][:3]
    rotations = np.zeros((channals_num, 3), dtype=np.float64)

    for i in range(channals_num):
        rotations[i] = motion_data[frame_id][3+3*i: 6+3*i]
    
    cnt = 0
    num_jnt = len(joint_name)
    joint_positions = np.zeros((num_jnt, 3), dtype=np.float64)
    joint_orientations = np.zeros((num_jnt, 4), dtype=np.float64)

    for i in range(num_jnt):
        if joint_parent[i] == -1: #root
            joint_positions[i] = root_position
            joint_orientations[i] = R.from_euler('XYZ', [rotations[cnt][0], rotations[cnt][1], rotations[cnt][2]], degrees=True).as_quat()
        else:
            if "_end" not in joint_name[i]:     # 末端没有CHANNELS
                cnt += 1
            r = R.from_euler('XYZ', [rotations[cnt][0], rotations[cnt][1], rotations[cnt][2]], degrees=True)
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * r).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(joint_orientations[joint_parent[i]]).apply(joint_offset[i])

    return joint_positions, joint_orientations

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    temp_motion_data = load_motion_data(A_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    
    # 计算 A_pose 中出现的数据
    A_name2index = {}
    cnt = 0
    for A_name in A_joint_name:
        if ("_end" not in A_name):
            cnt += 1
            A_name2index[A_name] = cnt
    # 计算 T_pose 中对应 A_pose 的顺序
    T_name2index = {}
    for T_name in T_joint_name:
        if ("_end" not in T_name):
            T_name2index[T_name] = A_name2index[T_name]
    
    # 得到 motioni_data
    motion_data = []
    for temp_line in temp_motion_data:
        line = []
        line.append(temp_line[0:3])
        for T_name in T_joint_name:
            if ("_end" not in T_name):
                line.append(temp_line[T_name2index[T_name] * 3 : T_name2index[T_name] * 3 + 3])
                if (T_name == "lShoulder"):
                    line[-1][-1] -= 45
                elif (T_name == "rShoulder"):
                    line[-1][-1] += 45
        motion_data.append(np.array(line).reshape(1,-1))
        cnt = 0
    motion_data = np.concatenate(motion_data, axis=0)

    return motion_data
