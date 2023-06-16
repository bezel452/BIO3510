from svm_c import work
import matplotlib.pyplot as plt

if __name__ == '__main__':
    accuracy = []
    seeds = []
    for i in range(0, 1001):
        accuracy.append(work(i))
        seeds.append(i)

    plt.plot(seeds, accuracy)
    plt.xlabel("Random_seed")
    plt.ylabel("Accuracy")

    plt.savefig("acc-seed.png")
    plt.show()

    print("平均准确率：", sum(accuracy) / len(accuracy))