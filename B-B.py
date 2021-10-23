import copy
import random
import sys
# from queue import PriorityQueue

import numpy as np
from docplex.mp.model import Model

# 数据
# c = np.array([3, 2])
# A = np.array([[2, 3], [4, 2]])
# b = np.array([14, 18])

# c = np.array([-1, 2,1,2])
# A = np.array([[1,1,-1,3],[0,1,3,-1],[-3,0,0,-1]])
# b = np.array([7,5,-2])

# c = np.array([23, 19,28,14,44])
# A = np.array([[8,7,11,6,19]])
# b = np.array([25])

c = np.array([10, 12, 7, 2])
A = np.array([[4, 5, 3, 1]])
b = np.array([10])


'''
m = Model(name='B&B')
# 变量
x = m.continuous_var_list(len(c), lb=0, name='x')
# x = m.integer_var_list(len(c), lb=0, name='x')
# 约束
cons = m.add_constraints(np.dot(A[i, :], x) <= b[i] for i in range(len(c)))
# 目标函数
obj = np.dot(c, x)
m.set_objective('max', obj)
sol = m.solve().get_values(x)
z = m.solve().get_objective_value()
# 参数
z_lb = -np.inf
z_ub = z
z_star = z

print(sol)
'''


class Node:
    def __init__(self, node_number):
        self.node_number = node_number
        self.consList = []
        self.sol = []
        self.z = 0
        self.z_lb = -np.inf
        self.z_ub = 0
        self.z_star = 0
        self.status = []
        self.pruned = False

    def solve_node(self, con=None):
        m = Model(name='B&B' + str(self.node_number))
        # 变量
        x = m.continuous_var_list(len(c), lb=0, name='x')
        # 约束
        cons = m.add_constraints(np.dot(A[i, :], x) <= b[i] for i in range(len(b)))
        if con is not None:
            self.consList.append(con)
            for con in self.consList:
                exec(con)
        # 目标函数
        obj = np.dot(c, x)
        m.set_objective('max', obj)
        # m.print_solution()
        solution = m.solve()
        if solution is None:
            self.pruned = True
            print("infeasible")
            # print(m.solve_details.gap)
        else:
            self.status.append(solution)
            self.sol = m.solve().get_values(x)
            self.z = m.solve().get_objective_value()
            # print(m.solve_details.columns)
            print(self.consList)


def is_integer(num):
    if num % 1 == 0:
        return True
    else:
        return False


if __name__ == "__main__":

    node0 = Node(0)
    node0.solve_node()
    node_list = []
    node_list_index = []
    node_number = 0
    node_list.append(node0)
    node_list_index.append(node_number)
    z_lb = -np.inf
    z_star = 0

    if all(is_integer(i) for i in node0.sol):
        z_lb = node0.z
        z_star = z_lb
        print('求得最优解:', node0.sol, z_star)
        sys.exit()
    if node0.pruned:
        print('此整数规划问题无解，因为其线性松弛问题无解')
        sys.exit()
    else:
        z_ub = node0.z
        print('求解出了初始解：', node0.sol, node0.z)

    node_list_index.remove(node_number)
    # 分两支：
    x_not_int = [(i, x) for i, x in enumerate(node0.sol) if not is_integer(x)]
    x_index = random.choice(x_not_int)
    x_b = x_index[1]
    x_lb = np.floor(x_b)
    x_ub = np.ceil(x_b)
    cons_lb = "m.add_constraint(x[{}]<={})".format(x_index[0], x_lb)
    cons_ub = "m.add_constraint(x[{}]>={})".format(x_index[0], x_ub)
    cons_all = [cons_lb, cons_ub]

    # 求解左右两分支节点
    for i in range(2):
        node_number += 1
        node_list_index.append(node_number)
        node_list.append(Node(node_number))
        node_list[node_number].solve_node(cons_all[i])
        print('初始化后，添加了条件：', cons_all[i])
        print('node_list_index 里面有：', node_list_index)
    print('node_list 长度为：', len(node_list))

    while len(node_list_index) != 0:
        # choice = random.choice(node_list_index)
        # print('随机选中节点', choice, '进行分支', '其线性松弛解为', node_list[choice].sol, node_list[choice].z)

        all_z = [(node_list[i].z, i) for i in node_list_index]
        all_z.sort(reverse=True)
        choice = all_z[0][1]
        print('选中z值最大的节点', choice, '进行分支','其线性松弛解为',node_list[choice].sol, node_list[choice].z)

        node_list_index.remove(choice)
        if node_list[choice].pruned:  # infeasibility prune
            print('node', choice, '无解')
            # node_list_index.remove(choice)
            print('node_list_index 里面有：', node_list_index)
            continue
        else:
            if node_list[choice].z < z_ub:
                z_ub = node_list[choice].z

        if node_list[choice].z < z_lb:  # bound prune
            print('node', choice, "的松弛解决（上界） 小于现有的下界，",z_lb," 没必要继续分支")
            # node_list_index.remove(choice)
            node_list[choice].pruned = True
            continue

        if all(is_integer(i) for i in node_list[choice].sol):  # optimality prune
            # node_list_index.remove(choice)
            node_list[choice].pruned = True

            if node_list[choice].z > z_lb:
                z_lb = node_list[choice].z
                # z_ub = node_list[choice].z
                print('找到可行解，设为下界！', node_list[choice].sol, node_list[choice].z)
                # node_list_index.remove(choice)
                print('node_list_index 里面有：', node_list_index)

        else:
            x_not_int = [(i, x) for i, x in enumerate(node_list[choice].sol) if not is_integer(x)]

            # x_index = random.choice(x_not_int)

            x_temp = [(abs(round(item[1]) + 0.5 - item[1]), item[0]) for item in x_not_int]
            x_temp.sort()
            for i in x_not_int:
                if i[0] == x_temp[0][1]:
                    x_index = i
                    break

            x_b = x_index[1]
            x_lb = np.floor(x_b)
            x_ub = np.ceil(x_b)
            cons_lb = "m.add_constraint(x[{}]<={})".format(x_index[0], x_lb)
            cons_ub = "m.add_constraint(x[{}]>={})".format(x_index[0], x_ub)
            cons_all = [cons_lb, cons_ub]

            # 求解左右两分支节点
            for i in range(2):
                node_number += 1
                temp_node = copy.deepcopy(node_list[choice])
                # temp_node = Node(node_number)
                temp_node.solve_node(cons_all[i])
                if temp_node.pruned:
                    temp_node.z = -np.inf
                if temp_node.z > z_lb:
                    print('temp_node.z > z_lb:',temp_node.z, ' > ', z_lb)
                    node_list_index.append(node_number)
                    node_list.append(temp_node)
                    print('求解节点', node_number, '并添加到node_list中')
                else:
                    node_list.append(0)

            print('node_list_index 里面有：', node_list_index)
            print('node_list 长度为：', len(node_list))



            # 求解左右两分支节点
            # for i in range(2):
            #     node_number += 1
            #     node_list_index.append(node_number)
            #     node_list.append(Node(node_number))
            #     node_list[node_number].solve_node(cons_all[i])
            #     print('求解节点', node_number, '并添加到node_list中')
            #     print('node_list_index 里面有：', node_list_index)
            # print('node_list 长度为：', len(node_list))

    # node1 = Node(1)
    # node1.solve_node(cons_lb)
    #
    # if not node0.pruned:
    #     print(node0.status)
    #     print(node0.sol)
    #     print(node0.z)
    #
    # node1 = Node(1)
    # node1.solve_node("m.add_constraint(x[1]>=3)")
    #
    # if not node1.pruned:
    #     print(node1.status)
    #     print(node1.sol)
    #     print(node1.z)

'''
1、构建 node，放进 nodelist[0]  增加一个数据node_numb=0
2、在 nodelist[0].sol 中 找到 most fractional 的 sol[i]
3、增加条件： cons = m.add_constraint(x[i] <= ceiling(sol[i]))  形成 node，放进nodelist[1], node_numb=1
4、增加条件： cons = m.add_constraint(x[i] >= floor(sol[i]))  形成 node，放进nodelist[2], node_numb=2
5、比较 node1 和 node2
    if node1.z_ub < node2.z_lb:
        del nodelist

'''
