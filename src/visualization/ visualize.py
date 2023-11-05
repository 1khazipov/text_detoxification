import matplotlib.pyplot as plt

def vizulize_blue(bleu_scores):

    # epochs
    indices = range(1, len(bleu_scores) + 1)

    plt.plot(indices, bleu_scores, marker='o', linestyle='-')
    plt.title('blue val')
    plt.xlabel('epoch')
    plt.ylabel('blue score')
    plt.grid(True)

    plt.show()

def vizualize_ter(ter_scores):
    # epochs
    indices = range(1, len(ter_scores) + 1)

    plt.plot(indices, ter_scores, marker='o', linestyle='-')
    plt.title('ter val')
    plt.xlabel('epoch')
    plt.ylabel('ter score')
    plt.grid(True)

    plt.show()

def vizualize_rouge1(rouge1_scores):
    # epochs
    indices = range(1, len(rouge1_scores) + 1)

    plt.plot(indices, rouge1_scores, marker='o', linestyle='-')
    plt.title('rouge1 val')
    plt.xlabel('epoch')
    plt.ylabel('rouge1 score')
    plt.grid(True)

    plt.show()

def vizualize_rouge2(rouge2_scores):
    # epochs
    indices = range(1, len(rouge2_scores) + 1)

    plt.plot(indices, rouge2_scores, marker='o', linestyle='-')
    plt.title('rouge2 val')
    plt.xlabel('epoch')
    plt.ylabel('rouge2 score')
    plt.grid(True)

    plt.show()