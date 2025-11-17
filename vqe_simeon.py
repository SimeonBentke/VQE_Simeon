import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


class base():
    def __init__(self):
        super().__init__()
        self.max_iter=1000
        self.store_history_energy=[]
        self.store_history_theta=[]
        self.store_history_qubit=[]

    def run(self):
        for _ in range(self.max_iter):
            self.store_history_energy.append(self.energy(self.state_update(self.theta)))
            self.store_history_theta.append(self.theta)
            self.store_history_qubit.append(self.state_update(self.theta).data)
            self.step_theta()
        self.plots()
    
    def state_update(self,theta):
        raise NotImplementedError("Implement this method in a subclass")

    def step_theta(self):
        self.theta=self.theta-self.step_size()*self.gradient()
    
    def plots(self):
        plt.title('theta(green) and energy(red) evolution')
        plt.plot(s.store_history_theta, 'g')
        plt.plot(s.store_history_energy, 'r')
        plt.show()


        x_vals=[x for x, y in s.store_history_qubit]
        y_vals=[y for x, y in s.store_history_qubit]
        plt.plot(y_vals)
        plt.plot(x_vals)
        plt.title('state-evolution')
        plt.show()




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
        print(gradient)
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
    
    def state_update(self,theta):
        return self.unitary(theta)@self.state
    
    def energy(self,state):
        energy=np.vdot(state,self.H @ state)
        return energy
    

class one_qubit_trivial_system_quantum(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.H=Operator(-X-Z)
        self.state=Statevector.from_instruction(QuantumCircuit(1))
        self.theta=np.pi/2
    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,s],[-s,c]])
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta)
        return self.state.evolve(u,qargs=[0])
    
    def energy(self,state):
        energy=np.vdot(state.data,self.H.data @ state.data)
        return energy
    

class one_dim_ising(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.H=Operator(X)
        self.state=Statevector.from_instruction(QuantumCircuit(1))
        #print(Statevector.from_instruction(self.state))
        self.theta=0
    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,-s],[s,c]])
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta)
        return self.state.evolve(u,qargs=[0])
    
    def energy(self,state):
        energy=np.vdot(state.data,self.H.data @ state.data)
        return energy

s=one_dim_ising()
s.run()




