import numpy as np
import matplotlib.pyplot as plt

class trplot():
    
    def __init__(self, transition_matrix, states = None, initial = None):

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

        if initial != None:
            self.initial = initial
        else: 
            self.initial = self.states[0]
        
        if self.initial not in self.states:
            raise ValueError("Initial must be contained in states.")


    def n_step_max_likelyhood(self, N, max_num_paths, plot_graph = False):
    
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        if not isinstance(max_num_paths, int) or N < 1:
            raise ValueError("Maximum number of paths must be an integer larger than 1.")

        path = [[self.initial]]
        probs = [1]
        state_to_index = {state: id for id, state in enumerate(self.states)}
    
        for num in np.arange(N):
            next_dict = {}
            num_valid_paths = len(path)
        
            for i in np.arange(num_valid_paths):
                current_state = path[i][num]
                current_probs  = probs[i]
                index = state_to_index[current_state]
                row_probs = self.transition_matrix[index]
                for k in np.arange(len(self.transition_matrix)):
                    if row_probs[k] != 0:
                        copy_path = np.copy(path[i])
                        next_path = np.append(copy_path, self.states[k])
                        next_prob = current_probs*self.transition_matrix[index,k]
                        next_dict[tuple(next_path)] = next_prob
            sorted_dict = sorted(next_dict.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_dict) <= max_num_paths:
                target_dict = sorted_dict[:len(sorted_dict)]
            else:
                target_dict = sorted_dict[:max_num_paths]
            path = [np.array(p) for p,_ in target_dict]
            probs = [v for _,v in target_dict]
        
        # Draw the trajectories
        if plot_graph == True:
            plt.figure(figsize = (8,6))
            num = 0
            for p in path:
                indexes = [state_to_index[s] for s in p]
                steps = range(N+1)
                plt.plot(steps, indexes, linestyle ='-',alpha=0.6, label="Trajectory {}".format(num) )
                num += 1

            plt.yticks(ticks=list(state_to_index.values()), labels=list(state_to_index.keys()))
            plt.legend()
            plt.xlabel("Steps")
            plt.ylabel("States")
            plt.title("{} Trajectories of Markov Chain with maximum likelyhood".format(max_num_paths))
            plt.grid()
            plt.show()

        return path, probs
        

    def n_step_random_trajectories(self, N, max_num_paths, plot_graph = False):

        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        if not isinstance(max_num_paths, int) or N < 1:
            raise ValueError("Maximum number of paths must be an integer larger than 1.")

        path = [np.array([self.initial]) for _ in np.arange(max_num_paths)]
        probs = list(np.ones(max_num_paths))
        state_to_index = {state: id for id, state in enumerate(self.states)}
    
        for num in np.arange(N):
            new_path = []
            new_prob = []
            for idx, p in enumerate(path):
                current_state = p[num]
                current_probs  = probs[idx]
                index = state_to_index[current_state]
                row_probs = self.transition_matrix[index]
                next_state = np.random.choice(self.states, p=row_probs)
                p = np.append(p,next_state)
                p_prob = probs[idx] * row_probs[state_to_index[next_state]]
                new_path.append(p)
                new_prob.append(p_prob)
            path = new_path.copy()
            probs = new_prob.copy()
        
        # Draw the trajectories
        if plot_graph == True:
            plt.figure(figsize = (8,6))
            num = 1
            for p in path:
                indexes = [state_to_index[s] for s in p]
                steps = range(N+1)
                plt.plot(steps, indexes, linestyle ='-', alpha=0.6, label="Trajectory {}".format(num) )
                num += 1

            plt.yticks(ticks=list(state_to_index.values()), labels=list(state_to_index.keys()))
            plt.legend()
            plt.xlabel("Steps")
            plt.ylabel("States")
            plt.title("{} random trajectories of Markov Chain".format(max_num_paths))
            plt.grid()
            plt.show()

        return path, probs



    def plot_eigenvalues(self, list_vectors, state_vector = None, name_vectors = None):
        
        idx = [id for id, _ in enumerate(self.states)]
        states = [state for _, state in enumerate(self.states)]
        if state_vector == None:
            state_vector = idx
        # if list_vectors.shape[1] != len(state_vector):
        #     raise ValueError("Length of vectors does not match.")
        if name_vectors != None:
            if not all(isinstance(x, str) for x in name_vectors):
                raise ValueError("Name vector must be list of strings.")

        index = 0
        plt.figure(figsize = (8,6))
        for vector in list_vectors:
            if name_vectors == None:
                plt.plot(idx, vector, linestyle='-', alpha=0.6, label='vector {}'.format(index))
            else:
                plt.plot(idx, vector, linestyle='-', alpha=0.6, label=name_vectors[index])
            index += 1
        plt.xticks(ticks=states, labels=idx)
        plt.legend()
        plt.xlabel("States")
        plt.ylabel("Probabilities")
        plt.title("Graph for vectors")
        plt.grid()
        plt.show()
        
            



    



                    

if __name__ == "__main__":
    from scipy.linalg import eig
    transition_matrix = [[0.9,0.1, 0, 0, 0],[0.6, 0, 0.4, 0, 0], [0, 0.6, 0, 0.4, 0], [0, 0, 0.6, 0, 0.4], [0, 0, 0, 1, 0]]
    initial = 2
    N = 50
    num = 3
    
    model = trplot(transition_matrix, initial=initial)
    path, probs = model.n_step_max_likelyhood(N, num, plot_graph=False)
    
    
    path, probs = model.n_step_random_trajectories(N, num, plot_graph=True)
    print(path, probs)              
 
    