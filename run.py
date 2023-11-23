from MR_Kmeans import MRKMeans
import os, sys, shutil, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from typing import List
from distances import euclidean_distance, cosine_distance, manhattan_distance


input_c = "centroids.txt"
CENTROIDS_FILE = "temp/centroids.txt"


def generate_data(filename: str):

    with open(filename, 'w') as file:
        if file.readlines() == []:
            for elem in np.random.randint(1000, size=(300,3)):
                x, y, z = elem
                print('\t'.join([str(x), str(y), str(z)]), file=file)


def get_centroids(job, runner) -> List:

    centroids = []
    for key, val in job.parse_output(runner.cat_output()):
        centroids.append(val)
    return centroids


def get_init_centroids(fname: str) -> List:

    centroids = []
    with open(fname, 'r') as f:        
        for line in f:
            x, y, z = line.split('\t')
            centroids.append([float(x), float(y), float(z)])
    return centroids


def write_centroids(centroids: List):

    with open(CENTROIDS_FILE,'w') as f:
        for c in centroids:
            f.write("%s\t%s\t%s\n"%(c[0],c[1],c[2]))


def make_res(centroids: List):

    with open('data.txt', 'r') as data_file, open('res.txt', 'w') as res_file:
        for line in data_file:
            x, y, z = line.split('\t')
            z = z.split('\n')[0]
            X  = np.asarray([float(x), float(y), float(z)])
            cl = np.argmin(mr_job.get_distances(X[np.newaxis, :], np.asarray(centroids)))
            res_file.write("%s\t%s\t%s\t%s\n"%(str(x), str(y), str(z), str(cl)))


def visualisation_3d(cluster_content: List):

    def get_label(elem):
        return int(elem[-1].split()[0])

    labels = []
    for elem in (cluster_content):
        labels.append(get_label(elem))

    colors = {0:'dodgerblue', 1:'yellow'}

    ax = plt.axes(projection="3d")
    plt.xlabel("x")    
    plt.ylabel("y")

    for label in np.unique(labels):
        x_coordinates = []
        y_coordinates = []
        z_coordinates = []
        for elem in cluster_content:
            if get_label(elem) == label:
                x_coordinates.append(int(elem[0]))
                y_coordinates.append(int(elem[1]))
                z_coordinates.append(int(elem[2]))
        ax.scatter(x_coordinates, y_coordinates, z_coordinates, c=colors[label])
    plt.savefig("cluster_res_visualization.png")


if __name__ == '__main__':
    
    args = sys.argv[1:]

    with open("data.txt", 'r') as file:
        if file.readlines() == []:
            generate_data("data.txt")

    shutil.copy(input_c, CENTROIDS_FILE)
    old_centroinds = sorted(get_init_centroids(input_c))

    ############################################################
    i=1 ########          run MR-pipeline             ##########
    ############################################################
    while True:

        print("Iteration #%i" % i)
        mr_job = MRKMeans(args=args + ['--c', CENTROIDS_FILE])

        with mr_job.make_runner() as runner:
            runner.run()
            new_centroids = get_centroids(mr_job,runner)
            new_centroids.sort()
            write_centroids(new_centroids)

        if new_centroids == old_centroinds:
            break
        else:
            old_centroinds = new_centroids
        print("======================================")
        i+=1
    ############################################################

    make_res(new_centroids)
    cluster_content = []
    with open("res.txt", 'r') as f:
        for line in f.readlines():
            cluster_content.append(line.split('\t'))
    visualisation_3d(cluster_content)
