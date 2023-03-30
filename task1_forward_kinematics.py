from viewer import SimpleViewer
import numpy as np
from Lab1_FK_answers import *
from viz import *

def part1(viewer, bvh_file_path):
    """
    part1 读取T-pose， 完成part1_calculate_T_pose函数
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    viewer.show_rest_pose(joint_name, joint_parent, joint_offset)
    viewer.run()


def part2_one_pose(viewer, bvh_file_path):
    """
    part2 读取一桢的pose, 完成part2_forward_kinematics函数
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    motion_data = load_motion_data(bvh_file_path)
    joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
    viewer.show_pose(joint_name, joint_positions, joint_orientations)
    viewer.run()


def part2_animation(viewer, bvh_file_path):
    """
    播放完整bvh
    正确完成part2_one_pose后，无需任何操作，直接运行即可
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    motion_data = load_motion_data(bvh_file_path)
    frame_num = motion_data.shape[0]
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0
        def update_func(self, viewer_):
            joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, self.current_frame)
            viewer.show_pose(joint_name, joint_positions, joint_orientations)
            self.current_frame = (self.current_frame + 1) % frame_num
    handle = UpdateHandle()
    viewer.update_func = handle.update_func
    viewer.run()


def part3_retarget(viewer, T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    Tips:
        我们不需要T-pose bvh的动作数据，只需要其定义的骨骼模型
    """
    # T-pose的骨骼数据
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    # A-pose的动作数据
    retarget_motion_data = part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path)

    #播放和上面完全相同
    frame_num = retarget_motion_data.shape[0]
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0
        def update_func(self, viewer_):
            joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, retarget_motion_data, self.current_frame)
            viewer.show_pose(joint_name, joint_positions, joint_orientations)
            self.current_frame = (self.current_frame + 1) % frame_num
    handle = UpdateHandle()
    viewer.update_func = handle.update_func
    viewer.run()



def fk(joint_name, joint_parent, joint_offset, motion_data_cur_frame):
    """
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    
    channals_num = (len(motion_data_cur_frame) -3) // 3
    root_position = motion_data_cur_frame[:3]
    rotations = np.zeros((channals_num, 3), dtype=np.float64)

    for i in range(channals_num):
        rotations[i] = motion_data_cur_frame[3+3*i: 6+3*i]
    
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

    return  root_position, joint_positions, joint_orientations


def ik():
    pass


def show_rest_pose(joint_name, joint_parent, joint_offset):
    num_jnt = len(joint_name)
    joint_positions = np.zeros((num_jnt, 3), dtype=np.float64)
    joint_orientations = np.zeros((num_jnt, 4), dtype=np.float64)
    #print(joint_name, joint_parent, joint_offset, length)
    for i in range(num_jnt):
        if joint_parent[i] == -1:
            joint_positions[i] = joint_offset[i]
        else:
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i]
        joint_orientations[i, 3] = 1.0
        #self.set_joint_position_orientation(joint_name[i], joint_positions[i], joint_orientations[i])
    return joint_positions


def norm_pos(ground_offset, root_pos_offset, motion_frame):
    motion_frame[:,:2] -=  root_pos_offset[:2]
    motion_frame[:,2] -=  ground_offset
    return motion_frame




def main():
    #bvh_file_path = "/home/demochan/repos/D1N9/Games105/lab1/data/walk60.bvh"
    #bvh_file_path = "data/A_pose_run.bvh"
    bvh_file_path = "/home/demochan/repos/D1N9/Games105/lab1/data/lafan1/walk1_subject5.bvh"
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    
    motion_data = load_motion_data(bvh_file_path)
    
    root_position, joint_positions, joint_orientations = fk(joint_name, joint_parent, joint_offset, motion_data[0])
    root_pos_init = root_position
    ground_init = np.min(joint_positions[:,2])  

    joints_seq = []

    for i in range(len(motion_data)):
        motion_frame = motion_data[i]
        motion_frame = norm_pos(ground_init,root_pos_init,motion_frame)
        

    
    '''
    # create a viewer
    viewer = SimpleViewer()
    #bvh_file_path = "data/lafan1/aiming1_subject1.bvh"
    bvh_file_path = "data/A_pose_run.bvh"

    # 请取消注释需要运行的代码
    #part1
    #part1(viewer, bvh_file_path)

    # part2
    part2_one_pose(viewer, bvh_file_path)
    #part2_animation(viewer, bvh_file_path)

    # part3
    part3_retarget(viewer, "data/walk60.bvh", "data/A_pose_run.bvh")
    '''

if __name__ == "__main__":
    main()