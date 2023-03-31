from viewer import SimpleViewer
import numpy as np
from fk_utils import *
from ik_utils import *
from viz import *
import tqdm


def get_path_from_root_to_end(joint_name, joint_parent, end_joint):
        """
        辅助函数，返回从root节点到end节点的路径
        
        输出：
            path: 各个关节的索引
            path_name: 各个关节的名字
        Note: 
            如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
            在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
            你可能会需要这两个输出。
        """
        
        # 从end节点开始，一直往上找，直到找到腰部节点
        path1 = [self.joint_name.index(self.end_joint)]
        while self.joint_parent[path1[-1]] != -1:
            path1.append(self.joint_parent[path1[-1]])
            
        # 从root节点开始，一直往上找，直到找到腰部节点
        path2 = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        
        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name = [self.joint_name[i] for i in path]
        return path, path_name, path1, path2
    

def norm_pos(ground_offset, root_pos_offset, motion_frame):
    motion_frame[:,:2] -=  root_pos_offset[:2]
    motion_frame[:,2] -=  ground_offset
    
    return motion_frame

def get_link(parent):
    link_lst = []
    for idx, idx_par in enumerate(parent):
        if idx_par == -1:
            continue
        link_lst.append([idx,idx_par])

    return link_lst

def main():
    #bvh_file_path = "/home/demochan/repos/D1N9/Games105/lab1/data/walk60.bvh"
    #bvh_file_path = "data/A_pose_run.bvh"
    
    #SK = 'LEFAN1'
    #num_joint = skel_dict[SK]['num_joint']
    #links =  skel_dict[SK]['links']
    bvh_file_path = "/home/demochan/repos/D1N9/Games105/lab1/data/lafan1/walk1_subject5.bvh"
    joint_name, joint_parent, joint_offset = calculate_T_pose(bvh_file_path)
    
    num_joint= len(joint_name)
    links = get_link(joint_parent)
    motion_data = load_motion_data(bvh_file_path)[:300]
    N,F = motion_data.shape

    root_position, joint_positions, joint_orientations = fk(joint_name, joint_parent, joint_offset, motion_data[0])
    root_pos_init = root_position
    ground_offset = np.min(joint_positions[:,2])  

    joints_seq = np.zeros((N,num_joint,3))

    motion_data = norm_pos(ground_offset, root_pos_init, motion_data)
    
    for i in tqdm.tqdm(range(len(motion_data))):
        #motion_data_cur_frame = norm_pos(ground_offset, root_pos_init, motion_data[i])
        motion_data_cur_frame = motion_data[i]
        _, joint_positions, joint_orientations = fk(joint_name, joint_parent, joint_offset, motion_data_cur_frame)
        if i == 0:
            joint_positions = np.reshape(joint_positions,(-1,3))
            viz_single(joint_positions)
        
        joints_seq[i] = joint_positions
    
    viz_anim(joints_seq, num_joint, links)

    
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