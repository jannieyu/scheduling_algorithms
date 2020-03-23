import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt





def init_solver(v, O_value=5):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    n = len(v) # rows

    # create array
    s = m.Array(m.Var, n)
    for i in range(n):
                s[i].value = 2.0
                s[i].lower = 0

    O = m.Var(value=O_value, lb=0)

    m.Equation(sum([int(v[i]) / s[i] + s[i] for i in range(len(v))]) == O)

    return m, s, O

def solver_results(s, m, O, verbose=True):

    m.Obj(O) # Objective
    m.options.IMODE = 3 # Steady state optimization
    m.solve(disp=False) # Solve

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i) + " " + str(s[i].value))
        print('Objective: ' + str(m.options.objfcnval))

    task_process_time = [float(1 / s[i].value[0]) for i in range(len(s))]

    return s, task_process_time


def make_assignment_visual(t, task_process_time, machine_task_list):

        t = create_start_end_times(t, task_process_time, machine_task_list)
       
        m = len(machine_task_list)
        # multi-dimensional data 
        machine_data = [[] for _ in range(m)]
        machine_labels = [[] for _ in range(m)]

        for i in range(m):
            machine_etfd = 0
            task_list = machine_task_list[i]
            for j in task_list:
                if machine_etfd < t[j][0]:
                    idle_time = t[j][0] - machine_etfd
                    machine_data[i].append(idle_time)
                    machine_labels[i].append('idle')
                process_time = task_process_time[j]
                machine_etfd = t[j][1]
                machine_data[i].append(process_time)
                machine_labels[i].append(str(j))

        segments = max([len(task_list) for task_list in machine_data])

        for i in range(len(machine_data)):
            for j in range(len(machine_data[i]), segments):
                machine_data[i].append(0)
                machine_labels[i].append('')

        data = []
        for s in range(segments):
            section = [machine_data[j][s] for j in range(m)]
            data.append(section)

        y_pos = np.arange(m)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        colors ='yg'
        patch_handles = []
        # left alignment of data task_starts at zero
        left = np.zeros(m) 
        for i, d in enumerate(data):
            patch_handles.append(ax.barh(y_pos, d, 
              color=colors[i%len(colors)], align='center', 
              left=left))
            left += d

        # search all of the bar segments and annotate
        for j in range(len(patch_handles)):
            for i, patch in enumerate(patch_handles[j].get_children()):
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = 0.5*patch.get_height() + bl[1]
                ax.text(x,y, machine_labels[i][j], ha='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.arange(m))
        ax.set_xlabel('Time')
        plt.show()



def create_start_end_times(t, task_process_time, machine_task_list):
    for m_lst in machine_task_list:
        earliest = 0
        for task in m_lst:
            if t[task][0] == None:
                t[task][0] = earliest
            t[task][1] = t[task][0] + task_process_time[task]
            earliest = t[task][0] + task_process_time[task]

    return t


def get_makespan(task_process_time, machine_task_list):
    makespan = 0
    for t in machine_task_list[1]:
        makespan += task_process_time[t]


    return makespan




# The following helper functions solve makepan + energy, not T + E

def init_solver_original(v, machine_task_list):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    n = len(v) # rows

    # create array
    s = m.Array(m.Var, n)
    for i in range(n):
                s[i].value = 2.0
                s[i].lower = 0

    P = m.Var(value=5, lb=0)
    m.Equation(sum([s[i] for i in range(len(v))]) == P)

    M1 = m.Var(value=5, lb=0)
    M2 = m.Var(value=5, lb=0)
    m.Equation(sum([1 / s[i]  for i in machine_task_list[0]]) == M1)
    m.Equation(sum([1 / s[i]  for i in machine_task_list[1]]) == M2)

    
    return m, s, P, M1, M2


def solver_results_original(m, s, P, M1, M2, verbose=True):

    m.Obj(P + m.max2(M1, M2)) # Objective
    m.options.IMODE = 3 # Steady state optimization
    m.solve(disp=False) # Solve

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i+1) + " " + str(s[i].value))
        print('Objective: ' + str(m.options.objfcnval))

    task_process_time = [float(1 / s[i].value[0]) for i in range(len(s))]

    return s, M1, M2, P,  task_process_time






# Example of using GEKKO

# m = GEKKO() # Initialize gekko

# # Use IPOPT solver (default)
# m.options.SOLVER = 3

# # Change to parallel linear solver
# m.solver_options = ['linear_solver ma97']

# # Initialize variables
# x1 = m.Var(value=1,lb=1,ub=5)
# x2 = m.Var(value=5,lb=1,ub=5)
# x3 = m.Var(value=5,lb=1,ub=5)
# x4 = m.Var(value=1,lb=1,ub=5)
# # Equations
# m.Equation(x1*x2*x3*x4>=25)
# m.Equation(x1**2+x2**2+x3**2+x4**2==40)
# m.Obj(x1*x4*(x1+x2+x3)+x3) # Objective
# m.options.IMODE = 3 # Steady state optimization
# m.solve(disp=False) # Solve
# print('Results')

# print('x1: ' + str(x1.value))
# print('x2: ' + str(x2.value))
# print('x3: ' + str(x3.value))
# print('x4: ' + str(x4.value))
# print('Objective: ' + str(m.options.objfcnval))


