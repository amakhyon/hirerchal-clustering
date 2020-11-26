import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import math

x_points = [2,2,8,5,7,6,1,4]
y_points = [10,5,4,8,5,4,2,9]
# x_points = [0.4,0.22,0.35,0.26,0.08,0.45]
# y_points = [0.53,0.38,0.32,0.19,0.41,0.3]

points = np.array([x_points,y_points])

def get_eucledian_distance(x1,x2,y1,y2):
  return round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2),2)

def init_distance_matrix(points):
  dimension = len(points[0])
  distance_matrix =  np.zeros((dimension,dimension))

  for row in range(dimension):
    # current_point is : X= points[0][row]   Y= points[1][row]
    x = points[0][row]; y = points[1][row]
    for column in range(dimension):
      tx = points[0][column];  ty = points[1][column]
      distance_matrix[row][column] = get_eucledian_distance(x,tx,y,ty)
  return distance_matrix


def get_min_in_matrix(matrix): #returns row and column of min distance
  dimension = len(matrix[0])
  mini =9999
  min = [1,1]
  for row in range(dimension):
    for column in range(dimension):
      if matrix[row][column] != 0 and matrix[row][column] < mini:
        mini = matrix[row][column]
        min[0] = row; min[1] = column
  return min

 

def get_clustered_matrix(distance_matrix,flag):
  mini = get_min_in_matrix(distance_matrix)
  cluster1_index = mini[0] #cluster 1 is where the new cluster will be placed because it has lower index
  cluster2_index = mini[1]
  dimension = len(distance_matrix[0]) - 1
  clustered_matrix =  np.zeros((dimension,dimension))
  if dimension == 2:

    for row in range(dimension):
      for column in range(dimension):
        if row != cluster1_index and row != cluster2_index: #if it is one of the unchanged clusters just copy it
          clustered_matrix[row][column] = distance_matrix[row][column]
        else:
          if flag =='min':
            clustered_matrix[row][column] = min(distance_matrix[cluster1_index][column], distance_matrix[cluster2_index][column])
          else:
            clustered_matrix[row][column] = max(distance_matrix[cluster1_index][column], distance_matrix[cluster2_index][column])
          
    print('\n')
    print('[updated matrix]')
    print(clustered_matrix)
    print('\n\n\n')
    return clustered_matrix

    
  else:

    for row in range(dimension):
      for column in range(dimension):
        if row != cluster1_index and row != cluster2_index: #if it is one of the unchanged clusters just copy it
          clustered_matrix[row][column] = distance_matrix[row][column]
        else:
            clustered_matrix[row][column] = min(distance_matrix[cluster1_index][column], distance_matrix[cluster2_index][column])
    print('cluster distance from all points')
    print(clustered_matrix[cluster1_index])
    
    print('\n')
    print('[updated matrix]')
    print(clustered_matrix)
    print('\n\n\n')
    return get_clustered_matrix(clustered_matrix,flag)








distance_matrix = init_distance_matrix(points)
print('=======================using min==================================')
m = get_clustered_matrix(distance_matrix,'min')
print('=======================using max==================================')
m = get_clustered_matrix(distance_matrix,'max')











# X = np.array([[2,10],
#               [2,5],
#               [8,4],
#               [5,8],
#               [7,5],
#               [6,4],
#               [1,2],
#               [4,9]
#               ])
# single = linkage(X, 'single')
# fig = plt.figure(figsize=(5,5))
# dn = dendrogram(single)
# complete = linkage(X, 'complete')
# fig = plt.figure(figsize=(5,5))
# dn = dendrogram(complete)
# plt.show()
# print(single[0])
