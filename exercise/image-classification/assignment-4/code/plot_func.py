# Author: LZS
# CreateTime: 2023/10/14  20:35
# FileName: plot_func
# Python Script

# plt绘制的函数

from matplotlib import pyplot as plt
from datetime import datetime


def plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, isSaveFig):
    current_time = datetime.now()  # 获取当前系统时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
    loss_path = './fig/fig-LOSS_' + formatted_time
    acc_path = './fig/fig-Accuracy_' + formatted_time

    fig = plt.figure(2)
    plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
    plt.legend(['Train Loss', 'Test Loss'], fontsize=14, loc='best')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid()
    if isSaveFig:
        plt.savefig(loss_path)
    # plt.show()

    fig = plt.figure(3)
    plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
    plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
    plt.legend(['Train Accuracy', 'Test Accuracy'], fontsize=14, loc='best')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.grid()
    if isSaveFig:
        plt.savefig(acc_path)
    # plt.show()


# def plot_curve(data, name, leg):
#     fig = plt.figure()
#     plt.plot(range(len(data)), data, 'blue')
#     plt.legend([leg], fontsize=14, loc='best')
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel(name, fontsize=14)
#     plt.grid()
#     plt.savefig('fig' + leg)
#     plt.show()


def plot_image(img, label, img_name, clas, figure_num):
    fig = plt.figure(figure_num)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # plt.tight_layout()
        plt.imshow(img[i].view(60, 60), cmap='gray')
        plt.title("{}:{}".format(img_name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('./fig/fig' + clas)
    plt.show()


def plot_samples(train, img_name='label', clas='60_60', figure_num=1):
    train_data09 = []
    train_label09 = []
    added_classes = set()
    for batch_data, batch_labels in train:
        for data, label in zip(batch_data, batch_labels):
            if label.item() not in added_classes:
                train_data09.append(data)
                train_label09.append(label)
                added_classes.add(label.item())
        if len(added_classes) == 10:
            break
    # 将train_data09和train_label09打包成一个元组的列表
    data_label_pairs = list(zip(train_data09, train_label09))

    # 按照标签值进行排序
    sorted_data_label_pairs = sorted(data_label_pairs, key=lambda x: x[1])

    # 将排序后的数据和标签重新解压缩到train_data09和train_label09中
    train_data09, train_label09 = zip(*sorted_data_label_pairs)

    # 转回为列表
    train_data09 = list(train_data09)
    train_label09 = list(train_label09)
    plot_image(train_data09, train_label09, img_name, clas, figure_num)

    return train_data09, train_label09
    # train_data09 = []
    # train_label09 = []
    # for i in range(10):  # 每个类型选一个数据
    #     train_data09.append(train_data[train_label == i][0])
    #     train_label09.append(torch.tensor(i))
    # print(len(train_data09), train_label09)
    # 10 [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4), tensor(5), tensor(6), tensor(7), tensor(8), tensor(9)]

    # plot_image(train_data09, train_label09, 'label', '60_60', 1)
