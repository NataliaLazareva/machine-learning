#Задача на бинарную классификацию, построение ROC-кривой.
import numpy as np
import matplotlib.pyplot as plt

def get_params(f_arr, b_arr, N, t):
    fotball_predicted = [0 if f_arr[i] < t else 1 for i in range(N)]
    basketball_predicted = [1 if b_arr[i] >= t else 0 for i in range(N)]
    FP = sum(fotball_predicted)
    TP = sum(basketball_predicted)
    TN = N - FP
    FN = N - TP
    alpha = FP / (FP + TN)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (2*N)
    return fotball_predicted, basketball_predicted, TN, TP,FN, FP, alpha, recall, accuracy

def get_other_params(FN, TP, FP, recall):
    beta = FN / (FN + TP)
    precision = TP / (TP + FP)
    f1_Score = 2 * precision * recall / (precision + recall)
    return beta, precision, f1_Score

def current_t(N, f_arr, b_arr, t):
    fotball_predicted, basketball_predicted, TN, TP, FN, FP, alpha, recall, accuracy = get_params(f_arr, b_arr, N, t)
    beta, precision, f1_Score = get_other_params(FN, TP, FP, recall)
    print("t:", t, '\n',"TP=", TP, "TN=", TN, "FP=", FP, "FN=", FN, '\n', "alpha: ", alpha, '\n'
          "beta: ", beta, '\n', "Precision: ", precision, '\n', "F1_Score: ", f1_Score, '\n', "Accuracy: ", accuracy)

def roc_cerve(f_arr, b_arr, N, t=300):
    accuracy_max=0
    t_optim=0
    alpha_mas = []
    recall_mas = []
    for i in range(t):
        fotball_predicted_cur, basketball_predicted_cur, TN, TP, FN, FP, alpha_cur, recall_cur, accuracy_cur = get_params(f_arr, b_arr, N, i)
        alpha_mas.append(alpha_cur)
        recall_mas.append(recall_cur)
        if accuracy_cur > accuracy_max:
            accuracy_max=accuracy_cur
            t_optim=i
    auc = get_auc_roc(alpha_mas, recall_mas)
    plt.figure()
    plt.plot(alpha_mas, recall_mas, 'b.-')
    plt.show()

    return accuracy_max, t_optim, auc

def get_auc_roc(alpha, recall):
    auc=0.0
    for i in range(len(recall)-1):
        sum = 0.0
        for j in range(len(alpha)-1):
                  sum += (alpha[j]-alpha[j+1])
        auc += sum*(recall[i] + recall[i + 1]) / 2
    return auc

def main():
 N = 1000
 mu_1 = 193
 sigma_1 = 7
 mu_0 = 180
 sigma_0 = 6
 t = 173
 # футболисты
 f_arr = np.random.randn(N) * sigma_0 + mu_0
 # баскетболисты
 b_arr = np.random.randn(N) * sigma_1 + mu_1

 current_t(N, f_arr, b_arr, t)

 accuracy_max, t_optim, auc = roc_cerve(f_arr, b_arr, N)
 print('accuracy_max:', accuracy_max, 'AUC ROC:', auc)
 current_t(N, f_arr, b_arr, t_optim)


if __name__ == '__main__':
    main()


