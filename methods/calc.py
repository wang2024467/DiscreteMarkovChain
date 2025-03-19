import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components


class EigenSol:
    '''
    Store the solutions for spectral decomposition of transition matrix
    '''
    def __init__(self, V, pi,  D, D_inv, eigval_V, eigvec_V, eigval_P, left_eigvec_P, right_eigvec_P):
        self.V = V
        self.pi = pi
        self.D = D
        self.D_inv = D_inv
        self.eigval_V = eigval_V
        self.eigvec_V = eigvec_V
        self.eigval_P = eigval_P
        self.left_eigvec_P = left_eigvec_P
        self.right_eigvec_P = right_eigvec_P


    def __repr__(self):
        return (f"EigenSolution(\n"
                f"  V={self.V},\n"
                f"  pi={self.pi}\n"
                f"  D={self.D},\n"
                f"  D_inv={self.D_inv},\n"
                f"  eigval_V={self.eigval_V},\n"
                f"  eigvec_V={self.eigvec_V},\n"
                f"  eigval_P={self.eigval_P},\n"
                f"  left_eigvec_P={self.left_eigvec_P},\n"
                f"  right_eigvec_P={self.right_eigvec_P},\n"
                f")")

class mccalc():
    
    def __init__(self, transition_matrix, states = None):
        '''
        Calculations in discrete Markov Chain.

        Parameters:
        transition_matrix: A probability matrix that indicates the transition probability for i state to j state.
        statas: The state vector, can be strings or numbers.
        '''

        self.transition_matrix = np.array(transition_matrix)
        if states != None:
            self.states = np.array(states)
        else:
            self.states = np.arange(len(transition_matrix))
        
        if len(self.states) != len(self.transition_matrix):
            raise ValueError("Length of states does not match size of transition matrix.")
        
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square.")
        
        if not np.allclose(self.transition_matrix.sum(axis=1),1):
            raise ValueError("Each row of transition matrix must sum to 1.")
        


    def transition_prob(self, N, current_state, final_state):
        '''
        Calculating the probability of transition from the current state to final state after N steps.
        '''
        
        if not isinstance(N, int) or N < 1:
            raise ValueError("Number of Iterations must be an integer greater than 1.")
        if current_state not in self.states:
            raise IndexError("Final state must be in state space.")
        if final_state not in self.states:
            raise IndexError("Final state must be in state space.")

        state_to_index = {state: id for id, state in enumerate(self.states)}
        p_N = np.linalg.matrix_power(self.transition_matrix, N)

        current_index = state_to_index[current_state]
        final_index = state_to_index[final_state]

        prob = p_N[current_index, final_index]
        return prob


    def homo_forward_Kolmogorov_dict(self, N, initial_probs):
        '''
        The probability distribution of all the states after N steps.
        '''

        if not isinstance(N, int) or N < 1:
            raise ValueError("Number of Iterations must be an integer greater than 1.")
        if len(initial_probs) != len(self.states):
            raise ValueError("Length of initial states does not match size of state space.")
        if not np.allclose(initial_probs.sum(),1):
            raise ValueError("Each row of transition matrix must sum to 1.")
        
        p_N = np.linalg.matrix_power(self.transition_matrix, N)
        final_probs = np.dot(initial_probs, p_N)
        final_dict = dict(zip(self.states, final_probs))
        return final_dict


    def homo_forward_Kolmogorov_prob(self, N, initial_probs, final_state):
        '''
        After computing the probability distribution for N steps, return the probability to transfer to the final state.
        '''

        if not isinstance(N, int) or N < 1:
            raise ValueError("Number of Iterations must be an integer greater than 1.")
        if len(initial_probs) != len(self.states):
            raise ValueError("Length of initial states does not match size of state space.")
        if not np.allclose(np.sum(initial_probs),1):
            raise ValueError("Each row of transition matrix must sum to 1.")
        if final_state not in self.states:
            raise IndexError("Final state must be in state space.")
        
        p_N = np.linalg.matrix_power(self.transition_matrix, N)
        final_probs = np.dot(initial_probs, p_N)
        final_dict = dict(zip(self.states, final_probs))
        return final_dict[final_state]
    

    def homo_backward_Kolmogorov_dict(self, N, initial_distribution = None):
        '''
        By assigning values on the states, find the expected value after N steps.
        '''

        if not isinstance(N, int) or N < 1:
            raise ValueError("Number of Iterations must be an integer greater than 1.")
        
        if initial_distribution == None:
            initial_distribution = self.states
            if not all(isinstance(x,(float, int)) for x in initial_distribution):
                raise ValueError("Initial distribution needs to be set.")
        else: 
            if not all(isinstance(x,(float, int)) for x in initial_distribution):
                raise ValueError("All elements in initial distribution must be either int or float.")
        
        p_N = np.linalg.matrix_power(self.transition_matrix, N)
        expected_value = np.dot(p_N, np.array(initial_distribution))
        final_dict = dict(zip(self.states, expected_value))
        return final_dict
    

    def homo_backward_Kolmogorov_prob(self, N, initial_state, initial_distribution = None):
        '''
        The probability of staying in the initial state after N steps.
        '''

        if not isinstance(N, int) or N < 1:
            raise ValueError("Number of Iterations must be an integer greater than 1.")
        
        if initial_state not in self.states:
            raise IndexError("initial state must be in state space.")

        if initial_distribution == None:
            initial_distribution = self.states
            if not all(isinstance(x,(float, int)) for x in initial_distribution):
                raise ValueError("Initial distribution needs to be set.")
        else: 
            if not all(isinstance(x,(float, int)) for x in initial_distribution):
                raise ValueError("All elements in initial distribution must be either int or float.")

        p_N = np.linalg.matrix_power(self.transition_matrix, N)
        expected_value = np.dot(p_N, np.array(initial_distribution))
        final_dict = dict(zip(self.states, expected_value))
        return final_dict[initial_state]
    

    def _if_irreducible(self):
        ''' 
        Whether the transition matrix is irreducible.
        '''

        adjoint = (self.transition_matrix > 0).astype(int)
        n_components, _ = connected_components(adjoint, directed=True, connection="strong")
        
        return n_components == 1 


    def _if_aperiodic(self):
        '''
        Whether the transition matrix is aperiodic.
        '''
        length = self.transition_matrix.shape[0]
        divisor_of_return = np.zeros(length, dtype=int)

        for i in range(length):
            trial = np.zeros(length, dtype=int)
            state = i
            for step in range(100):  
                state = np.random.choice(length, p=self.transition_matrix[state])
                trial[state] += 1
                if trial[state] > 1:
                    divisor_of_return[i] = np.gcd(divisor_of_return[i], step + 1)  

        return np.all(divisor_of_return == 1) 


    def compute_stationary_distribution(self, tol=1e-8, max_iters=2000):
        """
        Computing the stationary distribution of the transition matrix.
        """
        
        if self._if_irreducible() == False or self._if_aperiodic() == False:
            print(f"Warning: The stationary distribution may not be unique, irreducible: {self._if_irreducible()}; aperiodic: {self._if_aperiodic()}")

        length = self.transition_matrix.shape[0]
        pi = np.ones(length) / length 

        for i in range(max_iters):
            stationary = np.dot(pi, self.transition_matrix)  
            if np.linalg.norm(stationary - pi) < tol: 
                break
            pi = stationary  
        else:
            raise RuntimeError("The vector do not converge within max_iters.")
        
        check_point = np.dot(pi, self.transition_matrix) 
        if np.linalg.norm(check_point - pi) > tol:
            raise ValueError("The stationary distribution is computed in a wrong way.")

        return pi
    

    def examine_detailed_balance(self, stat_dist = None):
        """
        Examine whether transition matrix satisfies detailed balance
        
        Parameters:
        stat_dist: stationary distribution

        Returns: True or False
        """

        if stat_dist == None:
            stat_dist = self.compute_stationary_distribution()

        for i in range(len(stat_dist)):
            for j in range(len(stat_dist)):
                if not np.isclose(stat_dist[i] * self.transition_matrix[i, j], stat_dist[j] * self.transition_matrix[j, i], atol=1e-8):
                    return False
        return True
    

    def spectral_decomposition(self):
        """
        Spectral decomposition to find the eigenvalues, left and reight eigenvectors of transition matrix

        Ideas:
        1. Check transition matrix satisfies the detailed balance and compute the stationary distribution
        2. Set diagonal matrix D = diag(sqrt(pi))
        3. Let V = D P D^{-1}
        4. Find eigenvalues and eigenvectors of V
        5. Find left and right eigenvectors of P
        """
        length = self.transition_matrix.shape[0]
        
        # check detailed balance
        if self.examine_detailed_balance() == False:
            print("Warning: The transition matrix does not satisfy detailed balance.")
        
        # D is a diagonal matrix with square root of pi_j
        pi = self.compute_stationary_distribution()
        sqrt_pi = [np.sqrt(value) for value in pi]
        D = np.diag(sqrt_pi)

        #  V = D P D^{-1}
        D_inv = np.linalg.inv(D)
        V = np.dot(D, np.dot(self.transition_matrix, D_inv))

        # find eigenvalues and eigenvectors of V
        eigval_V, eigvec_V = np.linalg.eig(V)

        # find left and right eigenvectors of P
        eigval_P = eigval_V.copy()
        left_eigvec_P = np.dot(D, eigvec_V)
        right_eigvec_P = np.dot(D_inv, eigvec_V)

        index_eigen1 = None
        for index, val in enumerate(eigval_P):
            if np.isclose(val, 1.0, atol = 1e-8):
                index_eigen1 = index

        if index_eigen1 != None:
            if left_eigvec_P[0, index_eigen1] < 0:
                left_eigvec_P[:, index_eigen1] *= -1
            if right_eigvec_P[0, index_eigen1] < 0:
                right_eigvec_P[:, index_eigen1] *= -1
            if eigvec_V[0, index_eigen1] < 0:
                eigvec_V[:, index_eigen1] *= -1

        return EigenSol(V, pi, D, D_inv, eigval_V, eigvec_V, eigval_P, left_eigvec_P, right_eigvec_P)

    

    def check_stationary_with_eigenvector(self, eigenvector = None):
        '''
        Check whether the eigenvector related to deterministic eigenvalue lambda = 1 matches the stationary distribution.
        '''
        
        true_stationary_distribution = self.compute_stationary_distribution(tol = 1e-8)

        if eigenvector == None:
            sol = self.spectral_decomposition()
            eigvals = sol.eigval_P
            eigvecs = sol.left_eigvec_P
        
            index_eigen1 = None
            for index, val in enumerate(eigvals):
                if np.isclose(val, 1.0):
                    index_eigen1 = index
            
            if index_eigen1 == None:
                raise ValueError("No eigenvalue such that lambda = 1 is found.")
            
            eigenvector = eigvecs[:, index_eigen1]

        if np.linalg.norm(eigenvector - true_stationary_distribution) < 1e-6:
            return True
        return False






if __name__ == "__main__":

    transition_matrix = [[0.7,0.2, 0.1, 0, 0],[0.7, 0, 0.3, 0, 0], [0, 0.7, 0, 0.3, 0], [0, 0, 0.7, 0, 0.3], [0, 0, 0, 1, 0]]
    initial = 1
    N = 60
    num = 3

    model = mccalc(transition_matrix)
    prob = model.transition_prob(N,2,4)
    # print(prob)

    initial_probs = [0, 0, 1, 0, 0]
    final_state = 2
    prob = model.homo_forward_Kolmogorov_prob(N, initial_probs, 2)
    # print(prob)

    initial_probs = [0, 0, 1, 0, 0]
    final_state = 2
    probs = model.homo_forward_Kolmogorov_prob(N, initial_probs, final_state)
    print(prob)

    bool1 = model._if_irreducible()
    print(bool1)

    station = model.compute_stationary_distribution()
    print(station)

    check_detailed_balance = model.examine_detailed_balance()
    print(check_detailed_balance)

    sol = model.spectral_decomposition()
    print("V:\n", sol.V)
    print("D:\n", sol.D)
    print("D^-1:\n", sol.D_inv)
    print("The Eigenvalues of V:\n", sol.eigval_V)
    print("The Eigenvectors of V:\n", sol.eigvec_V)
    print("The Left Eigenvectors of P:\n", sol.left_eigvec_P)
    print("The Right Eigenvectors of P:\n", sol.right_eigvec_P)