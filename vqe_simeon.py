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
        theta_arr=np.array(self.store_history_theta)
        for i in range(self.theta.size):
            plt.plot(theta_arr[:,i], 'g')
            #plt.plot(self.store_history_energy, 'r')
        plt.title('theta(green) and energy(red) evolution')
        plt.show()


        #x_vals=[x for x, y in s.store_history_qubit]
        #y_vals=[y for x, y in s.store_history_qubit]
        #plt.plot(y_vals)
        #plt.plot(x_vals)
        #plt.title('state-evolution')
        #plt.show()




class step_size_const(base):
    def step_size(self):
        return 0.01


class gradiant_clas(base):
    def __init__(self):
        super().__init__()
        self.epsilon=1e-6
    def gradient(self):
        gradient=np.zeros(self.theta.size)
        for i in range(self.theta.size):
            state=self.state_update(self.theta)
            energy_state=self.energy(state)

            theta_epsilon=self.theta.copy()
            theta_epsilon[i]=self.theta[i]+self.epsilon
            state_epsilon=self.state_update(theta_epsilon)
            energy_state_epsilon=self.energy(state_epsilon)

            gradient[i]=(energy_state_epsilon-energy_state)/self.epsilon
        return gradient




class one_qubit_trivial_system(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.Hamilton=-X-Z
        self.state=np.array([1,0])
        self.theta=np.array([np.pi/2])

    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,s],[-s,c]])
        return u
    
    def state_update(self,theta):
        return self.unitary(theta[0])@self.state
    
    def energy(self,state):
        energy=np.vdot(state,self.Hamilton @ state)
        return energy
    

class one_qubit_trivial_system_quantum(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.Hamilton=Operator(-X-Z)
        self.state=Statevector.from_instruction(QuantumCircuit(1))
        self.theta=np.array([np.pi/2])
    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,s],[-s,c]])
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta[0])
        return self.state.evolve(u,qargs=[0])
    
    def energy(self,state):
        energy=np.vdot(state.data,self.Hamilton.data @ state.data)
        return energy



class one_dim_ising(gradiant_clas,step_size_const):
    def __init__(self):
        super().__init__()
        X=np.array([[0,1],[1,0]])
        Z=np.array([[1,0],[0,-1]])
        self.Hamilton=Operator(X)
        self.state=Statevector.from_instruction(QuantumCircuit(1))
        self.theta=np.array([0])
    def unitary(self,theta):
        c=np.cos(theta/2)
        s=np.sin(theta/2)
        u=np.array([[c,-s],[s,c]])
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta[0])
        return self.state.evolve(u,qargs=[0])
    
    def energy(self,state):
        energy=np.vdot(state.data,self.Hamilton.data @ state.data)
        return energy
    



class two_dim_ising_transversal_triv_ansatz(gradiant_clas,step_size_const):
    def __init__(self,J,H):
        super().__init__()
        H=np.array([[-J,-H,-H,0],[-H,J,0,-H],[-H,0,J,-H],[0,-H,-H,-J]])
        self.Hamilton=Operator(H)
        self.state=Statevector.from_instruction(QuantumCircuit(2))
        #print(Statevector.from_instruction(self.state))
        self.theta=np.array([0,np.pi])
    def unitary(self,theta):
        c1=np.cos(theta[0]/2)
        s1=np.sin(theta[0]/2)
        u1=np.array([[c1,-s1],[s1,c1]])

        c2=np.cos(theta[1]/2)
        s2=np.sin(theta[1]/2)
        u2=np.array([[c2,-s2],[s2,c2]])

        u=np.kron(u1,u2)
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta)
        return self.state.evolve(u)
    
    def energy(self,state):
        energy=np.vdot(state.data,self.Hamilton.data @ state.data)
        return energy
    


class n_dim_ising_triv_ansatz(gradiant_clas,step_size_const):
    def __init__(self,J,h):
        super().__init__()
        self.dim=J.shape[0]
        self.Hamilton=self.Ham(J,h)
        self.Hamilton=Operator(self.Hamilton)
        self.state=Statevector.from_instruction(QuantumCircuit(self.dim))
        self.theta=np.zeros(self.dim)
        for i in range(self.dim):
            self.theta[i]=np.pi/self.dim*i

    def Ham(self,J,h):
        Hamilton=np.zeros((2**self.dim,2**self.dim))
        for i in range(self.dim):
            Hamilton=Hamilton-h[i]*self.pauli_i_j(i,i)
            for j in range(i+1,self.dim):
                Hamilton=Hamilton-J[i,j]*self.pauli_i_j(i,j)
        return Hamilton


    def pauli_i_j(self,i,j):
        Id = np.array([[1,0],[0,1]])
        Z=np.array([[1,0],[0,-1]])
        tens_pauli=np.array([[1]])
        pauli_z_i_j = [Id for _ in range(self.dim)]
        pauli_z_i_j[i]=Z
        pauli_z_i_j[j]=Z
        for i in range(self.dim):
            tens_pauli=np.kron(tens_pauli,pauli_z_i_j[i])
        return tens_pauli


    def unitary(self,theta):
        u=np.array([[1]])
        for i in range(self.dim):
            u=np.kron(u,np.array([[np.cos(theta[i]/2), -np.sin(theta[i]/2)],[np.sin(theta[i]/2),  np.cos(theta[i]/2)]]))
        return Operator(u)
    
    def state_update(self,theta):
        u=self.unitary(theta)
        return self.state.evolve(u)
    
    def energy(self,state):
        energy=np.vdot(state.data,self.Hamilton.data @ state.data)
        return energy
    

#next steps:    finish transversal ising model      check
#               do general ising model              check
#               compare classic opti with QVE
#               ansatz eins nach oben schieben
#               better ansatz for ising
#               grad quant calculation



import numpy as np

# 10 spins
J = np.array([
    [0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [1.0,  0.0,  0.9,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.9,  0.0,  1.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  1.1,  0.0,  0.8,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.8,  0.0,  1.2,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  1.2,  0.0,  0.7,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.7,  0.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.9,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.9,  0.0,  1.1],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.1,  0.0],
], dtype=float)

h = np.array([0.3, -0.1, 0.0, 0.2, -0.25, 0.15, -0.05, 0.1, -0.2, 0.05], dtype=float)




# 7 spins
J = np.array([
    [0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [1.0,  0.0,  0.8,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.8,  0.0,  1.1,  0.0,  0.0,  0.0],
    [0.0,  0.0,  1.1,  0.0,  0.9,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.9,  0.0,  1.2,  0.0],
    [0.0,  0.0,  0.0,  0.0,  1.2,  0.0,  0.7],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.7,  0.0],
], dtype=float)

h = np.array([0.2, -0.1, 0.05, -0.15, 0.1, -0.05, 0.0], dtype=float)


#3 spins
#J = np.array([
#    [0.0,  1.2, -0.7],
#    [1.2,  0.0,  0.9],
#    [-0.7, 0.9,  0.0],
#], dtype=float)

#h = np.array([0.3, -0.2, 0.1], dtype=float)

s=n_dim_ising_triv_ansatz(J,h)
s.run()


#s.run()
#s=two_dim_ising(1,2)
#theta=np.array([np.pi/4,np.pi/4])
#print(s.unitary(theta))
#print(s.state_update(theta))
#print(s.H)




