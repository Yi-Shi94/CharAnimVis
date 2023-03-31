
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
def viz_single(x):
    print(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-100, +100)
    ax.set_ylim(-100, +100)
    ax.set_zlim(0, 200)
    ax.scatter(x[:,0], x[:,1], x[:,2], c='r',alpha=1)
    #print(x.shape)
    for i in range(x.shape[0]):

        ax.text(x[i,0], x[i,1], x[i,2], str(i), (0,1,0), fontsize=10, color='r')
    plt.show()


def viz_anim(x, num_joint, links, out_file_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    link_data = np.zeros((len(links),x.shape[0]-1,3,2))
    
    xini = x[0]
    root_x = xini[0,0]
    root_y = xini[0,1]
    
    ax.set_xlim(root_x-100, root_x+100)
    ax.set_ylim(root_y-100, root_y+100)
    ax.set_zlim(0, 200)
    link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,1],xini[ed,1]],[xini[st,2],xini[ed,2]],color='r')[0]
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
        #print('sad',data_lst.shape)
        root_x = x[num,0,0]
        root_y = x[num,0,1]
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,1],:])
            obj.set_3d_properties(data[2,:])
            
            ax.set_xlim(root_x-100, root_x+100)
            ax.set_ylim(root_y-100, root_y+100)
            ax.set_zlim(0, 200)
    
    line_ani = animation.FuncAnimation(fig, update_links, x.shape[0]-1, fargs=(link_data, link_obj),
                        interval=50, blit=False)

    if out_file_name is None:
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30) 
        line_ani.save(out_file_name, writer=writergif)