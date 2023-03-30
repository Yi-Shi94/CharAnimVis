
import matplotlib.pyplot as plt


def viz_single(x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-100, +100)
    ax.set_ylim(-100, +100)
    ax.set_zlim(0, 200)
    ax.scatter(x[:,0], x[:,1], x[:,2], c='r',alpha=1)
    plt.show()


def viz_anim(x, num_joint, links, out_file_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    if x.shape[-1] <= 3+12*num_joint and x.shape[-1]>=3+3*num_joint:
        dxdy = x[...,:2] 
        dr = x[...,2]
        x = np.reshape(x[...,3:3+3*num_joint],(-1,num_joint,3))
        
        cur_loc = np.array([[0.0,0.0,0.0]])
        traj = np.zeros((dxdy.shape[0],3))
        yaws = np.cumsum(dr)
        yaws = yaws - (yaws//(np.pi*2))*(np.pi*2)

        for i in range(1,x.shape[0]):
            cur_pos = np.zeros((1,3))
            cur_pos[0,0] = dxdy[i,0]
            cur_pos[0,2] = dxdy[i,1]
            cur_loc += np.dot(cur_pos,rot(yaws[i]))
            traj[i,:] = copy.deepcopy(cur_loc)
            x[i,:,:] = np.dot(x[i,:,:],rot(yaws[i])) + copy.deepcopy(cur_loc)

    elif x.shape[-1] == 66:
        x = np.reshape(x,(-1,num_joint,3))

    if x.shape[0] == 1:
        x = x[0]

    link_data = np.zeros((len(links),x.shape[0]-1,3,2))
    xini = x[0]
    
    link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color='r')[0]
                    for st,ed in links]

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    for i in range(1,x.shape[0]):
        
        for j,(st,ed) in enumerate(links):
            pt_st = x[i-1,st] #- y_rebase
            pt_ed = x[i-1,ed] #- y_rebase
            link_data[j,i-1,:,0] = pt_st
            link_data[j,i-1,:,1] = pt_ed
            
    def update_links(num, data_lst, obj_lst):
        #print(data_lst.shape)
        
        cur_data_lst = data_lst[:,num,:,:] 
        cur_root = cur_data_lst[4,:,0]
        root_x = cur_root[0]
        root_z = cur_root[2]
        
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,2],:])
            obj.set_3d_properties(data[1,:])
            
            ax.set_xlim(root_x-4, root_x+4)
            ax.set_zlim(0, 5)
            ax.set_ylim(root_z-4, root_z+4)
    
    line_ani = animation.FuncAnimation(fig, update_links, x.shape[0]-1, fargs=(link_data, link_obj),
                        interval=50, blit=False)

    if out_file_name is None:
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30) 
        line_ani.save(out_file_name, writer=writergif)