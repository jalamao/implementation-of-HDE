from test import GraphDrawing
from datetime import datetime


if __name__ == "__main__":

    graph_file = "grid_50x50.csv"  # "3elt.csv"; "grid_50x50.csv"; "grid_100x100_corners_connected.csv"; "tests.csv"; "tri_mesh_6"
    version = "PCA"
    start_time = datetime.now()
    gd = GraphDrawing(dimension=50, epsilon=0.01, version=version)  # default dimension is 50
    computation_time = 0
    plot_time = 0
    if graph_file == "3elt.csv":
        gd.transform('data/3elt.csv')
        end_computation = datetime.now()
        computation_time = (end_computation - start_time).seconds
        print("time of computation:", computation_time)
        gd.plot("figure_3elt.png")  # drawing image is slow when solving for large number (1000's) of points
        end_plot = datetime.now()
        plot_time = (end_plot - end_computation).seconds
        print("time of plot:", plot_time)

    if graph_file == "grid_50x50.csv":
        gd.transform('data/grid_50x50.csv')
        end_computation = datetime.now()
        computation_time = (end_computation - start_time).seconds
        print("time of computation:", computation_time)
        gd.plot("figure_grid_50x50.png")  # drawing image is slow when solving for large number (1000's) of points
        end_plot = datetime.now()
        plot_time = (end_plot - end_computation).seconds
        print("time of plot:", plot_time)

    if graph_file == "grid_100x100_corners_connected.csv":
        gd.transform('data/grid_100x100_corners_connected.csv')
        end_computation = datetime.now()
        computation_time = (end_computation - start_time).seconds
        print("time of computation:", computation_time)
        gd.plot(
            "figure_grid_100x100_corners_connected.png")  # drawing image is slow when solving for large number (1000's) of points
        end_plot = datetime.now()
        plot_time = (end_plot - end_computation).seconds
        print("time of plot:", plot_time)

    if graph_file == "tests.csv":
        gd.transform('data/tests.csv')
        end_computation = datetime.now()
        computation_time = (end_computation - start_time).seconds
        print("time of computation:", computation_time)
        gd.plot("figure_tests.png")  # drawing image is slow when solving for large number (1000's) of points
        end_plot = datetime.now()
        plot_time = (end_plot - end_computation).seconds
        print("time of plot:", plot_time)

    if graph_file == "tri_mesh_6":
        gd.transform("tri_mesh_6")
        end_computation = datetime.now()
        computation_time = (end_computation - start_time).seconds
        print("time of computation:", computation_time)
        gd.plot("figure_tri_mesh_6.png")
        end_plot = datetime.now()
        plot_time = (end_plot - end_computation).seconds
        print("time of plot:", plot_time)

    end_time = datetime.now()
    time = (end_time - start_time).seconds
    with open("cost_time.txt", "a") as f:
        f.write("version:" + version + "  " + graph_file + ": " + "time_of_computation=" + str(
            computation_time) + "  time_of_plot=" + str(plot_time) + "  total_time=" + str(time) + "\n")
    print("total time:", time)


