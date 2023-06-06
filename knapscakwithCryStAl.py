import numpy as np
import matplotlib.pyplot as plt

# Knapsack Problem
def knapsack_fitness(x, weights, values, capacity):
    total_weight = np.sum(x * weights)
    total_value = np.sum(x * values)
    if total_weight > capacity:
        #Toplam ağırlık kapasiteyi aştığında direkt çözümü sıfır yapıp katmamak yerine birim başına düşen değer hesaplayıp en vasat olan eşyayı çıkarabilirdik.
        fitness = 0
    else:
        fitness = total_value
    return fitness

# Boundary Handling / sınır işleme
def bound(x):
    return np.clip(x, 0, 1)

# Problem Information / problem bilgileri
weights = np.array([1, 2, 3, 4, 5, 10, 6, 7, 6, 5, 5, 2])  # Weight of each item / Her bir eşyanın ağırlığı
values = np.array([5, 4, 3, 2, 1, 8, 5, 6, 5, 4, 4, 3])  # Value of each item / Her bir eşyanın değeri
capacity = 30  # Knapsack capacity / Çantanın kapasitesi

# Algorithm Parameters / Algoritmanın Parametreleri
MaxIteration = 100
Cr_Number = 10

# Initialization / Başlatma
#0 ile 2 arasında (0-1) kristal sayısı kadar satır ağırlık dizisi kadar sutün
Crystal = np.random.randint(0, 2, size=(Cr_Number, len(weights)))
Fun_eval = np.zeros(Cr_Number)

for i in range(Cr_Number):
    Fun_eval[i] = knapsack_fitness(Crystal[i, :], weights, values, capacity)

# The best Crystal / En iyi kristal
idbest = np.argmax(Fun_eval)
Crb = Crystal[idbest, :]

# Number of Function Evaluations / Fonksiyon değerlendirme sayısı
Eval_Number = Cr_Number

# Search Process / Arama işlemi(süreci)
Iter = 0
Conv_History = np.zeros(MaxIteration)

while Iter < MaxIteration:
    for i in range(Cr_Number):
        # Generate New Crystals / Yeni kristalleri üretme
        Crmain = Crystal[np.random.permutation(Cr_Number)[0], :]
        RandNumber = np.random.permutation(Cr_Number)[0]
        RandSelectCrystal = np.random.permutation(Cr_Number)[:RandNumber]
        if len(RandSelectCrystal) != 1 :
            Fc = np.mean(Crystal[RandSelectCrystal, :], axis=0) 
        else:
            Fc = Crystal[RandSelectCrystal[0], :]

        r, r1, r2, r3 = 2 * np.random.rand(4) - 1

        # New Crystals / Yeni Kristaller
        NewCrystal = np.zeros((4, len(weights)))
        NewCrystal[0, :] = bound(Crystal[i, :] + r * Crmain)
        NewCrystal[1, :] = bound(Crystal[i, :] + r1 * Crmain + r2 * Crb)
        NewCrystal[2, :] = bound(Crystal[i, :] + r1 * Crmain + r2 * Fc)
        NewCrystal[3, :] = bound(Crystal[i, :] + r1 * Crmain + r2 * Crb + r3 * Fc)

        for i2 in range(4):
            # Evaluating New Crystals / Yeni kristallerin değerlendirilmesi
            Fun_evalNew = knapsack_fitness(NewCrystal[i2, :], weights, values, capacity)
            # Updating the Crystals / fitness değere göre kristallerin güncellenmesi
            if Fun_evalNew > Fun_eval[i]:
                Fun_eval[i] = Fun_evalNew
                Crystal[i, :] = NewCrystal[i2, :]
            # fonksiyon değerlendirme sayısı güncelleme
            Eval_Number += 1
    
    #Değerlendirilen kristalde çözümün fitness değeri
    #print(Fun_evalNew)
    
    # Her bir iterasyonda kristal çözümleri 
    idrandom = np.argwhere(Fun_eval)
    Crsolution = Crystal[idrandom,:]
    #print("Solution      :",Crsolution)
    
    #En iyi kristal
    #iterasyondaki en iyi fitness değerine sahip çözümü al
    idbest = np.argmax(Fun_eval)
    Crb = Crystal[idbest, :]
    BestCr = Crb
    Conv_History[Iter] = Fun_eval[idbest]
    #iterasyon değerlendirme
    print("Iteration",str(Iter) ,"Best Fitness =" ,str(Conv_History[Iter]))
    print("Best Solution :",BestCr)
    Iter += 1

Conv_History = Conv_History[1:Iter]

# Plot Convergence History / Yakınsama geçmişi grafiği çizdirme
plt.plot(Conv_History)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.show()
