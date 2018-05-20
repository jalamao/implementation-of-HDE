# coding:utf-8
def find_node_count(graph=None):
    tmp = []
    for i in graph:
        if i[0] not in tmp:
            tmp.append(i[0])
        if i[1] not in tmp:
            tmp.append(i[1])
    return max(tmp)

def transfer_file(file_name=None):
    input_f = open("tmp_data/" + file_name, "r")
    output_f = open("data/" + file_name, 'a')
    l = []
    # set_node = set()
    # edge = file.readlines()
    # for i in zip(edge):
    #     print(i)


    # # edge_count = edge.__len__()
    # # print(edge_count)
    # # for i in edge:
    #
    for i in input_f:
        i = i.split(sep=',')  # split string according to white space.
        ii = i[1].strip()
        l.append([int(i[0]), int(ii)])
        # set_node.add(int(i[0]), int(i[1]), int(i[2][0])
        # print(i[2][0])
    # edge_count = l.__len__()
    # node_count = find_node_count(l)
    # print("edges:", edge_count)
    # print("nodes:", node_count)
    # f.write(str(node_count) + "\n" + str(edge_count) + "\n")
    for i in l:
        output_f.write(str(i[0]) + "," + str(i[1]) + "," + str(1) + "\n")
        # f.write(str(i[0] - 1) + " " + str(i[1] - 1) + " " + str(i[2] - 1) + "\n")
    #     s1 = str(int(i[0]) - 1)
    #     s2 = str(int(i[1]) - 1)
    #     s3 = str(int(i[2]) - 1)
    #     f.write(s1 + " " + s2 + " " + s3 + "\n")
    input_f.close()
    output_f.close()










if __name__ == "__main__":

    file_name = "4elt.csv"
    transfer_file(file_name)

