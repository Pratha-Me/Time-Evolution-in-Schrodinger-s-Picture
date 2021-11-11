'''

In this python programming I have utilised CRANCK-NICHOLSON algorithm to devise an unitary
time operator comprised of the hamiltonian of system, afterwards time evolution was achieved
Moreover, the programming paradigm is 'Object-Oriented programing' abbreviated as 'OOP'
and not 'Procedural programming'. The particular technique deployed is 'encapsulation'
of method and fields/attributes.

Different shape and magnitude of potential bar can be implemented for extrapolation at line 60
by changing values of p1 and p2, and v altogether.

Thank you for your attention, any compliment or remark is welcomed at the end of the presentation.
Enjoy Learning and have a good day

'''

import sys
import numpy as np
from numpy.linalg import inv
from matplotlib import animation, patches as mp, pyplot as plt

class wave(object):

   #THE CONSTRUCTOR OF THE CLASS OR INITIALIZER IN PYTHON-------------
   def __init__(self,w,u,n):

      maplist = [w,u]

      # Both 'ini_w' and 'unit_optr' are '@PUBLIC variable/attribute/field'
      (self.ini_w, self.unit_optr) = map(np.asarray, maplist)

      assert self.ini_w.shape == (n,1)
      assert self.unit_optr.shape == (n,n)
   #------------------------------------------------------------------

   # THE TIME EVOLUTION IS CARRIED OUT INSIDE THIS METHOD.------------
   def evo_time(self):

      wt = np.matmul(self.unit_optr,self.ini_w)
      return (wt,np.absolute(wt))
   #------------------------------------------------------------------

   # THE SETTERS AND GETTERS METHODS ---------------------------------
   @property
   def wave_evo(self):
       # This Getter methods return type is 'void'.
       pass

   @wave_evo.setter
   def wave_evo(self,pre_y):
       self.ini_w = pre_y
   #------------------------------------------------------------------

# DECLARATION OF PHYSICAL PARAMETERS
n = 1350 # n =1000 NUMBER OF PARTITIONS OF SPACE
x=0.0; xo= 15.0; t= 1000; m =1.0; si= 1.5; l=60.; dx= l/float(n); a1,b1 = 25,25.5 #l =60., t =40000; $h=1.0$
p1,p2 = int(a1/dx),int(b1/dx); ko = 6.5; dt =0.01; v= (ko**2/2. + 1.0)  #dt =0.0001, ko =16.0 p1 =20, p2 =30
global z; z = np.linspace(0.,l,n); Id = np.identity(n)

# Construction of Initial **__nORMALIZED__** wavefunction with **__PHASE__**.
w = np.zeros((n,1), dtype = 'complex')
for i in range (0,n):
    w[i] = np.sqrt(np.sqrt((1/(np.pi*si**2))))*np.exp(-(x-xo)**2/(2*si**2))*np.exp(1.j*ko*x)
    x += dx

# *****************INITIAL WAVEFUNCTION CONSTRUCTED*******************
print ('The energy and potential height are %2.4f and %2.4f respectively' %(0.5*(ko**2 + 0.5/(si**2)), v))
print ('The norm of the initialised wavefunction is %2.2f' %(np.sqrt(np.absolute(np.vdot(w,w)))))

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

a = Id + ham     # ************* WE NEED IT'S INVERSE FOR ONCE BEFORE TIME LOOP ******************
b = Id - ham     # $$$$$$$$$$$$$ WE NEED A TIME EVOLUTION OPERATOR INVERSE(A)*B $$$$$$$$$$$$$$$$$$
a_inverse = inv(a)
''' In order to check if inverse was obtained or not following supressed 'print' syntax should be TRUE'''
#print(np.allclose(np.dot(a, a_inverse), np.eye(n)))

# THE 'UNITARY TIME OPERATOR' AS GIVEN BY $CRANCK-NICHOLSON ALGORITHM$
u = np.matmul(a_inverse,b)

# MAKE AN OBJECT-------------------------
evw = wave(w,u,n)
#----------------------------------------

''' First, set up the axes for the first impression and note that 'line,' will be updated in loop created by FuncAnimation.'''
fig = plt.figure()
ax = plt.axes(xlim=(0., l), ylim=(0., 0.9))

# Second, set up the figure, the axis, and the plot element we want to animate
center_line = ax.axvline(0, c='r', ls='-.')
center_nline = ax.axvline(0, c='r', ls='-.')
center_line.set_data(a1, [-1, 1])
center_nline.set_data(b1, [-1, 1])
#----------------------------------------------------------------------------------------------
plt.axvspan(a1, b1, 0., .8, facecolor='#FF7F50', alpha=0.7) ##c = '#2ca02c','#afeeee','#a5a391'#
descpt = mp.Patch(color='#FF7F50', label='The potential bar')
plt.legend(handles=[descpt],loc =1)
#----------------------------------------------------------------------------------------------
plt.ylabel('Norm of the wave $|\Psi_t (x)|$',fontsize=16, color = 'm',style='italic', fontweight='bold')
plt.xlabel('Direct Space $x$',fontsize=16, color = 'm',style='italic', fontweight='bold')
#----------------------------------------------------------------------------------------------

# Put title
plt.title('The time evolution of Gaussian Wave',fontsize=23, fontweight='bold',color='r')
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
'''Initialize the graph axes and Set the plotting line characteristics '''
line, = ax.plot([], [], 'k--.', lw =0.5)

''' initialization function: Initializes the values of line to be plotted (usually blank) however it will be updated during loop created by FuncAnimation. In FuncAnimation, if #blit = True#, then init will be retained and rest of the data will be updated. This eliminates all the redundant factor.'''
def init():
    line.set_data([], [])
    return line,

''' animation function.  This is called sequentially and will be called by FuncAnimation repeatedly
    in order to create the animation dictatedd by curve equation.'''
def animate(i):
    '''print(i) #$$$$$To check if animation.FuncAnimation is calling this animate function multiple times or not$$$$$'''
    (y,yt) = evw.evo_time()         # The Getter, wave_evo(self) is a void method, instead the computation is done by the written method.
    evw.wave_evo = y                # Calls Setter
    line.set_data(z, yt)
    return line,
#----------------------------------------------------------------------------------------------

# CALL THE ANIMATOR.  blit=true MEANS ONLY RE-DRAW THE PARTS THAT HAVE CHANGED.
anim = animation.FuncAnimation(fig, animate, init_func=init,\
                               frames=t, interval=30, blit=False)
#anim = anim.save('schrodinger_barrier.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.show()

#----------------------------------------------------------------------------------------------
sys.exit('TIMEOUT: MANIPULATE VARIABLES FOR DESIRED OUTCOME FOR THE NEXT ANIMATION')
