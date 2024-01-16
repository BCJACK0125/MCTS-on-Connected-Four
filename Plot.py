import matplotlib.pyplot as plt
import numpy as np

prob1 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.627, 0.253, 0.492, 0.998, 1.000])
prob2 = np.array([0.862, 0.727, 0.453, 0.406, 0.249, 0.501, 0.998, 1.000])
prob3 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.499, 0.999, 1.000])
prob4 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.627, 0.998, 1.000])
prob5 = np.array([0.862, 0.727, 0.453, 0.513, 0.996, 1.000])
prob6 = np.array([0.862, 0.727, 0.453, 0.406, 0.249, 0.501, 0.998, 1.000])
prob7 = np.array([0.862, 0.727, 0.999, 1.000])
prob8 = np.array([0.862, 0.998, 0.998,1.000])

prob9 = np.array([0.862, 0.727, 0.453, 0.406, 0.249, 0.000])
prob10 = np.array([0.862, 0.727, 0.453, 0.406, 0.249, 0.501, 0.000])
prob11 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.499, 0.000])
prob12 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.627, 0.253, 0.000])
prob13 = np.array([0.862, 0.727, 0.453, 0.406, 0.563, 0.627, 0.253, 0.492, 0.000])
prob14 = np.array([0.862, 0.727, 0.453, 0.513, 0.000])

index1 = np.arange(1, 11)
index7 = np.arange(1, 10)
index2 = np.arange(1, 9)
index3 = np.arange(1, 8)
index4 = np.arange(1, 7)
index5 = np.arange(1, 6)
index6 = np.arange(1, 5)


# 画出折线图
plt.plot(index6, prob8, color='red', linestyle='-')
plt.plot(index1, prob1, color='blue', linestyle='-')
plt.plot(index3, prob11, color='green', linestyle='-')
plt.plot(index4, prob9, color='gray', linestyle='-')
plt.plot(index2, prob2, color='blue', linestyle='-')
plt.plot(index2, prob3, color='blue', linestyle='-')
plt.plot(index2, prob4, color='blue', linestyle='-')
plt.plot(index4, prob5, color='blue', linestyle='-')
plt.plot(index2, prob6, color='blue', linestyle='-')
plt.plot(index6, prob7, color='blue', linestyle='-')
plt.plot(index3, prob10, color='gray', linestyle='-')
plt.plot(index2, prob12, color='green', linestyle='-')
plt.plot(index7, prob13, color='gray', linestyle='-')
plt.plot(index5, prob14, color='green', linestyle='-')

# 添加图例，蓝色为蓝方胜率，绿色为绿方胜率, 灰色为平局，红色为我方胜率
plt.legend(['Our Policy', 'Blue wins','Green wins', 'Draw',], loc='lower left')

plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.title('Probability of Winning')
plt.savefig("images/Covergence.png")
plt.show()
