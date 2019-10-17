'''

In this python programming I have utilised CRANCK-NICHOLSON algorithm to devise an unitary
time operator comprised of the hamiltonian of system, afterwards time evolution was achieved
Moreover, the programming paradigm is 'Object-Oriented programing' abbreviated as 'OOP'
and not 'Procedural programming'. The particular method deployed is widely known as
the 'setters and getters'

Different shape and magnitude of potential bar can be implemented for extrapolation at line 51
by changing values of p1 and p2, and v altogether.

Thank you for your attention, any compliment or remark is welcomed at the end of the presentation.
Enjoy Learning and have a good day

'''

import sys
import numpy as np
from numpy.linalg import inv
from matplotlib import animation, patches as mp, pyplot as plt

class wave(object):

# CONSTRUCTOR IN JAVA OR INITIALIZER IN PYTHON
   def __init__(self,w):

      # __ini_w is a '@PRIVATE variable/attribute/field'
      self.__ini_w = w
#------------------------------------------------

# SETTER AND GETTER IN PYTHON IS DONE BY DECORATOR '@'.
# THE COUNTERPART OF 'JAVA STYLE' PROGRAMMING FOR '@PRIVATE/PROTECTED/PUBLIC' TYPE OF VARIABLE.
   @property
   def wave_evo(self):
      return self.__ini_w

   @wave_evo.setter
   def wave_evo(self,pre_y):
      self.__ini_w = pre_y
#----------------------------------------------------------------------------------------

# DECLARATION OF PHYSICAL PARAMETERS
n = 1350 # n =1000 number of partitions of space
x=0.0; xo= 20.0; t= 1000; m =1.0; si= 1.5; l= 60.; dx= l/ float(n); a1,b1 = 30,32 #l =60., t =40000; $h=1.0$
p1,p2 = int(a1/dx),int(b1/dx); ko = 6.5; dt =0.01; v= (ko**2/2. - 1.0)  #dt =0.0001, ko =16.0 p1 =20, p2 =30
global z; z = np.linspace(0.,l,n); Id = np.identity(n)

# CONSTRUCTION OF INITIAL **__NORMALIZED__** WAVEFUNCTION WITH **__PHASE__**.
w_ini = np.zeros((n,1), dtype = 'complex')
for i in range (0,n):
    w_ini[i] = np.sqrt(np.sqrt((1/(np.pi*si**2))))*np.exp(-(x-xo)**2/(2*si**2))*np.exp(1.j*ko*x)
    x += dx
# *****************INITIAL WAVEFUNCTION CONSTRUCTED*******************

print ('The energy and potential height are %2.4f and %2.4f respectively' %(0.5*(ko**2 + 0.5/(si**2)), v))
print ('The norm of the initialised wavefunction is %2.2f' %(np.sqrt(np.absolute(np.vdot(w_ini,w_ini)))))

# MAKE AN INSTANCE OF THE CLASS WAVE
evw = wave(w_ini)
#-----------------------------------

# *****************HAMILTONIAN MATRIX CONSTRUCTION*******************
ham = np.zeros((n,n), dtype = 'complex')
j = 0
while j < n:

    ham[j,j] = (0.5*dt*1.j)/(dx**2)

    try:
        ham[j,j+1] = (0.5*dt*1.j)*(-1)*0.5/(dx**2)
        ham[j+1,j] = (0.5*dt*1.j)*(-1)*0.5/(dx**2)

    except IndexError:
        break

    j += 1

# 'ADDITION OF POTENTIAL IN HAMILTONIAN MATRIX', THE POTENTIAL IS SET BETWEEN A1 AND B1 ON X-AXIS.
for i in range(p1, p2):
   ham[i,i] += (0.5*dt*v*1.j)

a = Id + ham     # ************** WE NEED IT'S INVERSE FOR ONCE BEFORE TIME LOOP ******************
b = Id - ham     # $$$$$$$$$$$$$$ WE NEED A TIME EVOLUTION OPERATOR INVERSE(A)*B $$$$$$$$$$$$$$$$$$
a_inverse = inv(a)
u = np.matmul(a_inverse,b) # THE 'UNITARY TIME OPERATOR' AS GIVEN BY 'CRANCK-NICHOLSON ALGORITHM.'

''' In order to check if inverse was obtained or not following supressed 'print' syntax should be TRUE'''
#print(np.allclose(np.dot(a, a_inverse), np.eye(n)))

''' First, set up the axes for the first impression and note that 'line,' will be updated in loop created by FuncAnimation.'''
fig = plt.figure()
ax = plt.axes(xlim=(0., l), ylim=(0., 0.9))

# Second, set up the figure, the axis, and the plot element we want to animate
center_line = ax.axvline(0, c='r', ls='-.')
center_nline = ax.axvline(0, c='r', ls='-.')
center_line.set_data(a1, [-1, 1])
center_nline.set_data(b1, [-1, 1])
#----------------------------------------------------------------------------------------------
plt.axvspan(a1, b1, 0., .8, facecolor='#9B7FB0', alpha=0.7) ##c = '#2ca02c','#afeeee','#a5a391'#
descpt = mp.Patch(color='#9B7FB0', label='The potential bar')
plt.legend(handles=[descpt],loc =1)
#----------------------------------------------------------------------------------------------
plt.ylabel('Norm of the wave $|\Psi_t (x)|$',fontsize=16, color = 'b',style='italic')#, fontweight='bold')
plt.xlabel('Direct Space $x$',fontsize=16, color = 'b',style='italic')#, fontweight='bold')
#----------------------------------------------------------------------------------------------

# Put title
plt.title('The time evolution of Gaussian Wave',fontsize=23, fontweight='bold',color='b')
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
'''Initialize the graph axes and Set the plotting line characteristics '''
line, = ax.plot([], [], 'k--.', lw =0.5)

''' initialization function: Initializes the values of line to be plotted (usually blank) however it will be updated during loop created by FuncAnimation. In FuncAnimation, if #blit = True#, then init will be retained and rest of the data will be updated. This eliminates all the redundant factor.'''
def init():
    line.set_data([], [])
    return line,

''' animation function.  This is called sequentially and will be called by FuncAnimation repeatedly in order to create the animation dictatedd by curve equation.'''
def animate(i):
    '''print(i) #$$$$$To check if animation.FuncAnimation is calling this animate function multiple times or not$$$$$'''
    yt = np.matmul(u,evw.wave_evo)   # Calls Getter
    evw.wave_evo = yt                # Calls Setter
    line.set_data(z, np.absolute(yt))
    return line,
#----------------------------------------------------------------------------------------------

# CALL THE ANIMATOR.  BLIT=TRUE MEANS ONLY RE-DRAW THE PARTS THAT HAVE CHANGED.
anim = animation.FuncAnimation(fig, animate, init_func=init,\
                               frames=t, interval=30, blit=False)
#anim = anim.save('schrodinger_barrier.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

plt.show()

#----------------------------------------------------------------------------------------------
sys.exit('TIMEOUT: MANIPULATE VARIABLES FOR DESIRED OUTCOME FOR THE NEXT ANIMATION')
