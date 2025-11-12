import numpy as np
import matplotlib.pyplot as plt

class base():
    def __init__(self):
        super().__init__()
        self.max_iter=1000
        self.store_history_energy=[]
        self.store_history_theta=[]

    def run(self):
        for _ in range(self.max_iter):
            self.store_history_energy.append(self.energy(self.state_update(self.theta)))
            #self.store_history_theta.append(self.theta)
            self.step_theta()

    def state_update(self,theta):
        return self.unitary(theta)@self.state

    def step_theta(self):
        self.theta=self.theta-self.step_size()*self.gradient()

class step_size_const(base):
    def step_size(self):
        return 0.01

class gradiant_clas(base):
    def __init__(self):
        super().__init__()
        self.epsilon=1e-6
    def gradient(self):
        state=self.state_update(self.theta)
        energy_state=self.energy(state)

        theta_epsilon=self.theta+self.epsilon
        state_epsilon=self.state_update(theta_epsilon)
        energy_state_epsilon=self.energy(state_epsilon)

        gradient=(energy_state_epsilon-energy_state)/self.epsilon
        return gradient




class one_qubit_trivial_system(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.H=-X-Z
        self.state=np.array([1,0])
        self.theta=np.pi/2
    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,s],[-s,c]])
        return u
    def energy(self,state):
        energy=np.vdot(state,self.H @ state)
        return energy

s=one_qubit_trivial_system()
s.run()

plt.plot(s.store_history_theta, 'g')
plt.plot(s.store_history_energy, 'r')
plt.show()
