import numpy as np
from numpy import pi
from scipy.integrate import odeint
import random as rd
from random import randint
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('seaborn-paper')
from matplotlib import animation

def find_nearest(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# initializing animation elements
def init():
    line.set_data([], [])
    time_text.set_text('')
    reward_text.set_text(reward_template % total_r[i])
    return line, time_text, reward_text

# line and text
def animate(t):
    thisx = [0, x[t]]
    thisy = [0, y[t]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (t * dt))
    return line, time_text  #

# mapping from theta to the x,y-plane
def pendulum_plot(l, theta):
    x = -l * np.sin(theta)
    y = l * np.cos(theta)
    return x, y


# inverse pendulum dynamics
def inverse_pendulum(xt, t, at):
    g = 9.81  # gravity
    b = 0.0  # dissipation
    tau = at  # torque
    theta, dot_theta = xt
    ddot_theta = tau + g * np.sin(theta) - b * dot_theta
    dot_x = dot_theta, ddot_theta
    return dot_x

def pos_sqrt(a):
	root = np.multiply(np.sign(a),np.sqrt(abs(a)))
	return root

fig_format = '.pdf'

# action-space discretization
tau_tile = np.array([0, -2, 2])

# state-space discretization
dot_theta_tile = np.linspace(10, -10, 63)
theta_tile = np.linspace(-pi, pi, 126)

# desired state
theta_d = 0;

# weights for reward function
w_tau = 0
w_theta = 1
w_dot_theta = 0.25
weights = [w_tau, w_theta ,w_dot_theta]

# start from already existing data (load = True)
load = False
# intialize q-table non-zero
preset = False
# q-table
if os.path.isfile('qvalue_inverted_pendulum.npy') and load:
    q = np.load('qvalue_inverted_pendulum.npy')
else:
    # intialize q-table which is a 3D-zero-matrix
    q = np.ones([len(tau_tile), len(theta_tile), len(dot_theta_tile)])
    if preset == True:
        # preset the q-table with rewards
        for i in range(0, len(theta_tile)):
            q[:, i, :] = -w_theta*((abs(theta_tile[i]) - theta_d) ** 2)
        for i in range(0, len(dot_theta_tile)):
            q[:, :, i] = q[:, :, i] - w_dot_theta * ((dot_theta_tile[i]) ** 2)
        q[find_nearest(0, tau_tile), find_nearest(theta_d, theta_tile), find_nearest(0, dot_theta_tile)] = 0

# episode
if os.path.isfile('start_episode.npy') and load:
    start_episode = np.load('start_episode.npy')
else:
    start_episode = 0

# eligibility matrix
e = np.zeros([len(tau_tile), len(theta_tile), len(dot_theta_tile)])

# step size
dt = 0.05

# time in seconds
ttime = 60

# simulation steps
steps = int(ttime / dt)

# learning episodes
num_episodes = 1000

#choose algorithm (0 = QLearning, 1 = SARSA)
algorithm = 0
# qlearning parameters
gamma = 0.99  # discount factor    
alpha = 1  # learning rate
lamda = 0.5 # decay rate eligibility matrix
epsilon = 0.5
eps_decay = 0.98
# print np.load('reward_episode.npy')
a = np.zeros([num_episodes - start_episode])

# intialize reward 
if os.path.isfile('reward_episode.npy') and load:
    total_r = np.append(np.load('reward_episode.npy'), np.zeros([num_episodes - start_episode]))
else:
    total_r = np.zeros(num_episodes - start_episode + 1)

# reward function
def reward(w, xt, at, dt):
    theta = xt[0]
    dot_theta = xt[1]
    tau = at
    r = -(w[1]*(abs(theta) - theta_d)**2 + w[2]*dot_theta**2 + w[0]*tau**2)
    return r


# observation - state prediction
def observation(xt_, at, dt):
    tt = [0, dt]
    xt = odeint(inverse_pendulum, xt_, tt, args=(at,))
    # map theta to [-pi,pi]
    if xt[-1, 0] > pi:
        xt[-1, 0] -= 2 * pi
    elif xt[-1, 0] < -pi:
        xt[-1, 0] += 2 * pi
    return xt[-1, :]


# policy - which action to take dependend on state
def policy(q, xt_, A, dt, eps):
    # if eps=0 -> greedy policy, exploitation
    # compare to random fp
    if rd.random() > eps:
        # greedy action
        k = find_nearest(xt_[0], theta_tile)
        l = find_nearest(xt_[1], dot_theta_tile)
        a = A[np.argmax(q[:, k, l])]
    else:
        # random action
        a = A[randint(0, len(A) - 1)]
    return a


# q-table 3D array
def q_learning(xt_, xt, at_, r, q, e, gamma, alpha, lamda):
    # find the nearest representation of state and action in the q-table
    j_ = find_nearest(at_, tau_tile)
    k_ = find_nearest(xt_[0], theta_tile)
    l_ = find_nearest(xt_[1], dot_theta_tile)
    k = find_nearest(xt[0], theta_tile)
    l = find_nearest(xt[1], dot_theta_tile)
    e[j_, k_, l_] = 1
    q += alpha * (r + gamma * max(q[:, k, l]) - q[j_, k_, l_]) * e
    e = gamma * lamda * e
    return q, e

def sarsa(xt_, xt, at_, at, r, q, e, gamma, alpha, lamda):
    # find the nearest representation of state and action in the q-table
    j_ = find_nearest(at_, tau_tile)
    k_ = find_nearest(xt_[0], theta_tile)
    l_ = find_nearest(xt_[1], dot_theta_tile)
    j = find_nearest(at, tau_tile)
    k = find_nearest(xt[0], theta_tile)
    l = find_nearest(xt[1], dot_theta_tile)
    e[j_, k_, l_] = 1
    q += alpha * (r + gamma * q[j, k, l] - q[j_, k_, l_]) * e
    e = gamma * lamda * e
    return q, e

# simulation
for i in range(start_episode, num_episodes + 1):
    # initial values
    xt_ = [pi, 0.0]
    yt = np.zeros((steps, len(xt_)))
    at_ = policy(q, xt_, tau_tile, dt, epsilon*eps_decay**i)
    rew = []

    print 'episodes left', num_episodes - i
    for j in range(0, steps):
        if algorithm == 0: #QLearning
            # take action
            at_ = policy(q, xt_, tau_tile, dt, epsilon*eps_decay**i)  # this could later be a neural network

            # make observation
            xt = observation(xt_, at_, dt)  # this can also be a neural network

            # get reward for the state transition
            r = reward(weights, xt, at_, dt)

            # update q-function
            q, e = q_learning(xt_, xt, at_, r, q, e,  gamma, alpha, lamda)

            #store state
            xt_ = xt

        elif algorithm == 1: #SARSA
            # make observation
            xt = observation(xt_, at_, dt)  # this can also be a neural network

            # get reward for the state transition
            r = reward(weights, xt_, at_, dt)

            # take action
            at = policy(q, xt, tau_tile, dt, epsilon*eps_decay**i)  # this could later be a neural network
            
            #update q-function
            q, e = sarsa(xt_, xt, at_, at, r, q, e,  gamma, alpha, lamda)

            #store state and action
            at_ = at
            xt_ = xt

        if abs(xt[1])>10:
        	terminate = j
        	print 'termiated at time: ', j*dt
        	break
        # store state trajectory
        yt[j, :] = xt
        # store reward
        rew.append(r)

    # store average reward/step
    total_r[i] = sum(rew) / len(rew)

    # plot and data storage
    if i % 50 == 0:
        # reward plot
        plt.close('all')
        plt.figure()
        plt.plot(total_r[0:i:1])
        plt.ylim([min(total_r)-0.5,0])
        plt.title('learning curve')
        plt.ylabel(r'avg. reward / step')
        plt.xlabel(r'$n$ episodes')
        plt.text(-1,i/10,r'\alpha ='+str(alpha))
        plt.savefig('total_reward.'+fig_format)
        plt.close('all')

    if i % 100 == 0:    
        # q-table plot
        plt.close('all')
        plt.figure()
        #plt.imshow(np.transpose(np.maximum(np.maximum(q[0, :, :], q[1, :, :]), q[2, :, :])), aspect='auto',
         #          cmap='jet')
        plt.imshow(np.transpose(q.max(axis=0)), aspect='auto',
                  cmap='jet')
        plt.colorbar()
        plt.yticks(np.linspace(0, len(dot_theta_tile) - 1, 3), ['10', 0, '-10'])
        plt.xticks(np.linspace(0, len(theta_tile) - 1, 3), ['$-\pi$', 0, '$\pi$'])
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title(r'Action-Value-Table - $\max_{a_t}{Q(x_t,a_t)}$')
        plt.savefig('q_table'+str(i)+fig_format)
        plt.close('all')
        
        # plot the optimal action for a certain state
        plt.close('all')
        plt.figure()
        plt.imshow(np.transpose(q.argmax(axis=0)), aspect='auto', cmap=mpl.colors.ListedColormap(['purple', 'red', 'blue']), vmin=0,vmax=len(tau_tile))
        plt.yticks(np.linspace(0, len(dot_theta_tile) - 1, 3), ['10', 0, '-10'])
        plt.xticks(np.linspace(0, len(theta_tile) - 1, 3), ['$-\pi$', 0, '$\pi$'])
        plt.xlabel(r'$\theta$')
        plt.text(115,4,r'$\tau_{max}$',color='blue')
        plt.text(115,7,r'$\tau_{min}$',color='red')
        rect = patches.Rectangle((113.5,1.5),10,6.5,linewidth=1,edgecolor='black',facecolor='white',fill=True)
        # Add the patch to the Axes
        currentAxis = plt.gca()
        currentAxis.add_patch(rect) 
        #plt.colorbar.set_tickslabel(['$-\tau_{max}', 0, '$\tau_{min}$'])
        plt.ylabel(r'$\dot{\theta}$')
        plt.title(r'$Policy - a_t \sim \pi(x_t)$',fontsize = 12)
        plt.savefig('q_table_actor'+str(i)+fig_format)
        plt.close('all')
    if i % 100 == 0:
        # animation
        plt.close('all')
        [x, y] = pendulum_plot(1, yt[:, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        ax.set_aspect('equal')
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        reward_template = 'avg. reward / step = %.2f'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        reward_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',
        right='off',
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',labelleft='off') # labels along the bottom edge are off
        plt.title('Epsiode '+str(i))
        ani = animation.FuncAnimation(fig, animate, np.arange(0, len(rew)),
                                      interval=dt * 100, blit=False, init_func=init)
        ani.save('TD0_Q_learning_Episode_' + str(i) + '.mp4', fps=1 / dt)

        #  data storage
        np.save('qvalue_inverted_pendulum', q)
        np.save('start_episode', i)
        np.save('reward_episode', total_r[0:i:1])
