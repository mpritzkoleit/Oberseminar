'''
Author: Max Pritzkoleit
Institution: Technical University Dresden
last edit: 20/Jan/2017
'''
import numpy as np
from numpy import pi
from scipy.integrate import odeint
import random as rd
from random import randint
import sympy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('seaborn-paper')
from matplotlib import animation
import tensorflow as tf

def plot_reward():
    print 'reward plot'
    # reward plot
    plt.close('all')
    plt.figure()
    plt.plot(total_r[0:i:1])        
    plt.title('learning curve')
    plt.ylabel(r'avg. cost / step')
    plt.xlabel(r'$n$ episodes')
    plt.savefig(framework+'/total_cost'+fig_format)

def plot_animate():
    # initializing animation elements
    def init():
        line.set_data([], [])
        torque.set_data([], [])
        wheel.center = (0, 0)
        ax.add_patch(wheel)
        time_text.set_text('')
        reward_template = 'total cost = %.2f'
        reward_text.set_text(reward_template % total_r[i])
        return line, time_text, reward_text, wheel, torque
        print 'saving animation'

    # line and text
    def animate(t):
        thisx = [x_cart[t], x_tip[t]+x_cart[t]]
        thisy = [0.08, y_tip[t]+0.08]
        if rew[t]==0:
            line.set_color('g')
            wheel.set_color('g')
        else:
            line.set_color('k')
            wheel.set_color('k')
        line.set_data(thisx, thisy)
        torque.set_data([u[t]/max(abs(f_tile)), 0], [-0.05, -0.05])
        wheel.center = (thisx[0], 0.08)
        time_text.set_text(time_template % (t * dt))
        return line, time_text, wheel, torque

    # mapping from theta and s to the x,y-plane
    def cart_pole_plot(l, xt):
        x_tip = l * np.sin(xt[:,0])
        x_cart = xt[:,2]
        y_tip = l * np.cos(xt[:,0])
        return x_tip, y_tip, x_cart
    
    # animation
    [x_tip, y_tip, x_cart] = cart_pole_plot(l, yt)
    fig = plt.figure()
    fig.dpi = 150
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-(s_limit+l), s_limit+l), ylim=(-0.1, l*1.5))
    plt.title('Cart-Pole Regluator - Episode '+str(i)) 
    ax.set_aspect('equal')
    rail, = ax.plot([-1.2*s_limit, 1.2*s_limit], [0,0],'ks-')
    torque, = ax.plot([], [], '-', color='r', lw=4)
    line, = ax.plot([], [], 'o-', color='k')
    wheel = plt.Circle((0, 1), 0.08, color='k', fill=False, lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    reward_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
    ani = animation.FuncAnimation(fig, animate, np.arange(0, j), interval=dt * 100, blit=False, init_func=init)
    if learn_controller:
        ani.save(framework+'/videos/NFQ_episode' + str(i) + '.mp4', fps=1/dt)
    else:
        ani.save(framework + '/videos/Experiment_' + str(i) + '.mp4', fps=1 / dt)

def store_data():
    #store neural network
    if learn_controller:
        save_path = saver.save(sess, framework + '/data/agent_network.ckpt')
        # store data
        np.save(framework + '/data/total_reward', total_r[0:i:1])
        np.save(framework + '/data/dataset', D)
        print("Model saved in file: %s" % save_path)
    else:
        # store data
        np.save(framework + '/data/reward_experiments', total_r[0:i:1])
        np.save(framework + '/data/dataset_experiments', D)
        np.save(framework + '/data/xt', yt)
        np.save(framework + '/data/ut', u)
        np.save(framework + '/data/reward', rew)

# cartpole dynamics
def cart_pole(xt, t, at):
    f = at # input
    theta, dot_theta, s, dot_s = xt   
    x2, x4, x1, x3 = xt   
    # Systemparameter
    b = 0.2 # m
    c = 0.3 # m
    r = 0.05 # m
    m_w = 0.5 # kg
    m_p = 3.0 # kg
    J_w = 0.5*m_w*r**2 # kg*m2
    J_p = 1/12*m_p*(b**2+c**2) # kg*m2
    g = 9.81 # m/s2

    q_dot = np.array([x3, x4])
    
    # motion dynamics in matrix form
    M = np.array([[2*m_w + 2*J_w/(r**2) + m_p, m_p*c*np.cos(x2)],
                        [m_p*c*np.cos(x2), J_p + m_p*c**2]])
    C = np.array([[0, -m_p*c*np.sin(x2)*x4], [0, 0]])
    G = np.array([0, -m_p*c*g*np.sin(x2)])
    F = np.array([2/r*f, 0])
    

    xdot = np.dot(np.linalg.inv(M),(F-np.dot(C,q_dot)-G)) 
    dot_xt = [x4, xdot[1], x3, xdot[0]]
    return dot_xt

# reward function
def reward_nfq(xt, at, steps):
    theta = xt[0]
    dot_theta = xt[1]
    s = xt[2]
    dot_s = xt[3]
    f = at
    #if abs(s)>=s_limit or abs(dot_theta)>=dot_theta_max or abs(theta)>pi/2:
    if abs(s)>=1.1*s_limit or abs(theta)>pi/2:
        r = 1
        terminate = True
    elif abs(theta)<=theta_d and abs(s)<=s_d:
        r = 0
        terminate = False
    else:
        r = 0.01
        terminate = False
    #r = (8.7*abs(s)**2 + 8.7*(abs(theta) - theta_d)**2 + 0.00001*dot_theta**2 + 0.00001*dot_s**2 + 4.8*f**2)
    return r, terminate
 
# observation - state prediction
def observation(xt_, at, dt):
    tt = [0, dt]
    xt = odeint(cart_pole, xt_, tt, args=(at,))
    # map theta to [-pi,pi]
    if xt[-1, 0] > pi:
        xt[-1, 0] -= 2 * pi
    elif xt[-1, 0] < -pi:
        xt[-1, 0] += 2 * pi
    return xt[-1, :]

# take greedy action
def action(xt_,i):
    #epsilon-greedy policy
    if ((rd.random() >= max(epsilon,eps_decay**i)) or i > training_episodes ):
        # greedy action
        #determine neural network inputs to evalute which action generates the minimum cost
        nn_ff = np.ones([len(f_tile), 5])
        nn_ff[:,0] = f_tile
        nn_ff[:,1] = xt_[0]
        nn_ff[:,2] = xt_[1]
        nn_ff[:,3] = xt_[2]
        nn_ff[:,4] = xt_[3]
        at_ = f_tile[(sess.run(predict_argmin, feed_dict={nn_in: nn_ff}))]
    else:
        # random action
        at_ = f_tile[randint(0, len(f_tile) - 1)]
    return at_

def train_qnetwork():
    nn_predict = np.zeros([len(f_tile), 5])
    #train network on the whole data-set D
    if q_iteration:
        print 'training the Q-network on',len(D),'/',batch_size,'samples'
    else:
        print 'training the Q-network on',len(D),'samples'
    artificial_states = len(D)/100
    #training goal
    y_network = np.zeros([artificial_states+len(D),1])
    # (a,x)
    nn_input = np.zeros([artificial_states+len(D),5])
    #predict min(b)_Q(s',b)
    
    for k in range(0,len(D)):
        nn_input[k,0] = D[k][0] # at_
        nn_input[k,1] = D[k][1] # xt_[0]
        nn_input[k,2] = D[k][2] # xt_[1]
        nn_input[k,3] = D[k][3] # xt_[2]
        nn_input[k,4] = D[k][4] # xt_[3]
        #calculate the goal for selected transition
        nn_predict[:,0] = f_tile
        nn_predict[:,1] = D[k][5] # xt[0]
        nn_predict[:,2] = D[k][6] # xt[1]
        nn_predict[:,3] = D[k][7] # xt[2]
        nn_predict[:,4] = D[k][8] # xt[3]

        #set goal to r if goal-state or termination state is reached
        if D[k][9] == 1 or D[k][9] == 0:
            y_network[k,0] = np.matrix(D[k][9])
        else:
            y_network[k,0] = np.matrix(D[k][9] + gamma*sess.run(predict_min, feed_dict={nn_in: nn_predict}))
    
    #generate artifical state tranistions -> hint-to-goal heuristic
    for p in range(0,artificial_states/len(f_tile)):
        for k in range(0,len(f_tile)):
            nn_input[len(D)+k+len(f_tile)*p,0] = f_tile[k]
            nn_input[len(D)+k+len(f_tile)*p,1] = 0 # xt_[0]
            nn_input[len(D)+k+len(f_tile)*p,2] = 0 # xt_[1]
            nn_input[len(D)+k+len(f_tile)*p,3] = 0 # xt_[2]
            nn_input[len(D)+k+len(f_tile)*p,4] = 0 # xt_[3]
            y_network[len(D)+k+len(f_tile)*p,0] = 0
    #actual training of the neural network
    for _ in range(0,300):
        sess.run(optimizer, feed_dict={nn_in: nn_input, y_goal: y_network})  

def store_training_data():
    # should maybe be implemented as a dictionary instead of an array of arrays
    if [at_, xt_[0], xt_[1], xt_[2], xt_[3], xt[0], xt[1], xt[2], xt[3], r] not in D:
            D.append([at_, xt_[0], xt_[1], xt_[2], xt_[3], xt[0], xt[1], xt[2], xt[3], r])

    # delete transition from D if length of D is limited, FIFO principle
    if q_iteration == True and len(D)>batch_size:
        D.pop(0)

fig_format = '.pdf'


# action-space discretization
f_tile = np.array([0.1,-0.1,1,-1,2.5,-2.5])
#f_tile = np.array([-3,3])
s_max = 0.5
s_limit = 2*s_max
dot_theta_max = 10
#goal states
theta_d = 0.02
s_d = 0.05

# start from already existing data (load = True)
load = False

# intialize q-table non-zero
preset = False

# step size
dt = 0.01
# time in seconds for learning
ttime = 6

# time in seconds for controller performance evaluation
ttime_performance = 3

# simulation steps
steps = int(ttime / dt)

# learning episodes
training_episodes = 300
#total episodes
num_episodes = training_episodes

l = 0.5
# qlearning parameters
gamma = 0.99  # discount factor
epsilon = 0.1 # 0 -> greedy
eps_decay = 0

#neural network parameters
nhidden1 = 20
nhidden2 = 20

batch_size = 100*steps

#sample random transitions from data-set D?
use_batch_algorithm = False

#limit size of dataset to batchsize?
q_iteration = False

num_states = 4
num_actions = len(f_tile)

#nn_in = tf.placeholder(tf.float32, [None, num_states]) #None -> any length )
nn_in = tf.placeholder(tf.float32,[None, 5]) # nn_in(1) = at, nn_in(2:5) = xt
y_goal = tf.placeholder(tf.float32,[None, 1])

#weights
w1 = tf.Variable(tf.random_uniform([5,nhidden1],minval=-0.5,maxval=0.5))
w2 = tf.Variable(tf.random_uniform([nhidden1,nhidden2],minval=-0.5,maxval=0.5))
wout = tf.Variable(tf.random_uniform([nhidden2,1],minval=-0.5,maxval=0.5))

#biases
b1 = tf.Variable(tf.random_uniform([nhidden1],minval=0,maxval=0))
b2 = tf.Variable(tf.random_uniform([nhidden2],minval=0,maxval=0))
bout = tf.Variable(tf.random_uniform([1],minval=0,maxval=0))

#batch normalization 
nn_max = tf.reduce_max(nn_in,axis=0)
nn_min = tf.reduce_min(nn_in,axis=0)
nn_mean = tf.reduce_mean(nn_in,axis=0)
#max values for normalization purpuses
nn_in_min = tf.Variable(tf.zeros([1, 5]))
nn_in_max = tf.Variable(tf.ones([1, 5]))
nn_in_mean = tf.Variable(tf.random_normal([1, 5]))

# neural network
h1out = tf.nn.tanh(tf.add(tf.matmul(nn_in, w1),b1))
h2out = tf.nn.tanh(tf.add(tf.matmul(h1out, w2),b2))
y_out = tf.nn.sigmoid(tf.add(tf.matmul(h2out, wout),bout))
#h1out = tf.nn.relu(tf.add(tf.matmul(nn_in, w1),b1))
#h2out = tf.nn.relu(tf.add(tf.matmul(h1out, w2),b2))
#y_out = tf.nn.relu(tf.add(tf.matmul(h2out, wout),bout))
#y_out = tf.matmul(nn_in, W) + b
predict_max = tf.reduce_max(y_out)
predict_min = tf.reduce_min(y_out)
predict_argmax = tf.argmax(y_out, 0)
predict_argmin = tf.argmin(y_out, 0)

#loss function
loss = tf.square(y_goal-y_out, name = 'loss')
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_goal * tf.log(y_out), reduction_indices=[1]))
#optimization
#optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
#optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)
#reward data
total_r = np.zeros(num_episodes + 1)

#data-sets for storing transition [at_, xt_, xt, r]
D = []

#D = np.load('cart_pole/data/dataset.npy')
#launch model
sess = tf.InteractiveSession()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#restore saved model?
restore_model = False

#terminal state reached?
terminate = False

#set framework to select working_folder
framework = 'cart_pole'


#create an operation to initialize the variables
tf.global_variables_initializer().run()

# switch between training and evaluation
learn_controller = False 

if not(learn_controller) or restore_model:
    #load cart_pole regulator
    saver.restore(sess, framework+'/data/agent_network.ckpt')
    D = np.load('cart_pole/data/dataset.npy')
    print('Model and Data restored.')


# main loop
if learn_controller:
    for i in range(0, num_episodes + 1):
        #initialization
        #starting state
        xt_ = np.array([np.random.uniform(low=-30./180.*np.pi,high=30./180.*np.pi), 0.0, np.random.uniform(low=-.8,high=.8), 0.0], dtype = np.float32)
            #xt_ = np.array([pi, 0.0, 0.0, 0.0], dtype = np.float32)
        at_ = f_tile[randint(0, len(f_tile) - 1)]

        #state trajectory for animation
        yt = np.zeros((steps, len(xt_)))
        #reward trajectory
        rew = []

        #control trajectory
        u = []

        print 'episodes left', num_episodes - i

        #episode
        for j in range(0, steps):
            # take epsilon-greedy action
            at_ = action(xt_, i)

            # make observation (system simulation using ODE-solver)
            xt = observation(xt_, at_, dt)

            # get reward(or cost) for the state transition
            r, terminate = reward_nfq(xt, at_, steps)

            #store transitions
            store_training_data()

            #end of episode or terminal state reached -> train neural network
            if (j == steps-1 or terminate) and i < training_episodes and len(D)>steps or use_batch_algorithm:
                train_qnetwork()
            #store state
            xt_ = xt
            # store state trajectory
            yt[j, :] = xt

            u.append(at_)

            rew.append(r)

            if terminate:
                print 'termiated at: ',j*dt,' s'
                break
            if r==0:
                success_point = j
                print 'goal reached at: ', j*dt, ' s'

        total_r[i] = sum(rew) / len(rew)

        # plot and data storage
        if i % 10 == 0:
            plot_reward()

        if i % 10 == 0:
            plot_animate()

        if i % 10 == 0:
            store_data()
else:
    xt0 = np.array([[20./180.*np.pi, 0, 0, 0],[40./180.*np.pi, 0, 0, 0],[0, 0, 0.3, 0],[0, 0, 0.7, 0],[30./180.*np.pi, 0, 0.5, 0]])
    #xt0 = np.array([[np.pi/4, 0, 0, 0]])
    num_exp = len(xt0)
    for i in range(0, num_exp):
        print 'running experiment ', i
        xt_ = xt0[i]

        at_ = f_tile[randint(0, len(f_tile) - 1)]
        # state trajectory for animation
        yt = np.zeros((steps, len(xt_)))
        # reward trajectory
        rew = []

        # control trajectory
        u = []

        for j in range(0, int(ttime_performance/dt)):
            # take epsilon-greedy action
            at_ = action(xt_, 1000)

            # make observation (system simulation using ODE-solver)
            xt = observation(xt_, at_, dt)

            # get reward(or cost) for the state transition
            r, terminate = reward_nfq(xt, at_, steps)

            # store state
            xt_ = xt
            # store state trajectory
            yt[j, :] = xt

            u.append(at_)

            rew.append(r)

        total_r[i] = sum(rew) / len(rew)

        # plot and data storage

        plot_animate()

        store_data()
