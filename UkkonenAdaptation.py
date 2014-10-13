#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math, random
import fileinput

class Chain:
	# Класс описывает "цепочку". В данном случае просто последовательность цифр.
    def __init__(self, elements, idP='cent', reference=None): 
        self.elements = elements 	# элементы цепочки 
        self.idP = idP 				# ID цепочки. Сейчас, вероятнее всего, не используеся 
        self.n = len(elements) 		# длина цепочки
        self.reference = reference	# служебное поле
    def __repr__(self):
        return str(self.idP) + ' - ' + str(self.elements)

class Point:
	# Класс описывает точку в многомерном пространстве.
    def __init__(self, coords, idP='cent', reference=None):
        self.coords = coords
        self.idP = idP
        self.dim = len(coords)
        self.reference = reference
    def __repr__(self):
        return str(self.idP) + ' - ' + str(self.coords)

class Cluster:
	# Класс описывает кластер. Набор точек в многомерном пространстве или цепочек.
    def __init__(self, points,distanceFunc):
        self.points = points #точки, которые принадлежат данному кластеру
        if len(points) > 0: #проверяем, не пустой ли кластер. Если пустой, то центройд считать не надо
        	self.centroid = self.calculateCentroid(distanceFunc)
    def __repr__(self):
        return str(self.points)
    def update(self, points, distanceFunc):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid(distanceFunc)
        return distanceFunc(old_centroid, self.centroid)
    def calculateCentroid(self,distanceFunc):
    	# ниже расчет центройдов для двух случаев. 
    	# если кластер содержит обычные точки, то считается просто средняя точка, как в обычной геометрии
		if self.points[0].__class__.__name__ == "Point":
			dim = self.points[0].dim
			reduce_coord = lambda i:reduce(lambda x,p : x + p.coords[i],self.points,0.0)
			centroid_coords = [reduce_coord(i)/len(self.points) for i in range(dim)]
			return Point(centroid_coords)

		# если кластер содержит цепочки, то мы ищем медойд (т.е. элемент, сумма расстояний от которого до остальных элементов кластера минимальна)
		if self.points[0].__class__.__name__ == "Chain":
			minDist =10000
			for p1 in self.points:
				dist = 0
				for p2 in self.points:
					dist = dist + distanceFunc(p1,p2)
				if minDist>dist:
					minDist = dist
					centroid = p1
		return centroid

def kmeans(points, k, cutoff, distanceFunc):
	# Один из основных методов программы. Алгоритм кластеризации к-средних. Основные комментарии будут ниже, по тексту самого метода
	# Входящие параметры
	# points - точки или цепочки, на которых проходит кластеризация
	# k - набор кластеров
	# cutoff - чтобы избежать бесконечных циклов используется значение cutoff. Если перемещение центройдов на итерации меньше этого значения, то мы завершаем кластеризацию
	# distanceFunc - функция расстояния, которую будем использовать при кластеризации
    

    # инициализацию проводим пока в "тепличном режиме", а именно, зная что точки идут у нас в нужном порядке, мы: 
    # первый центройд берем из первой трети объектов, второй центройд из второй трети объектов, и третий - из третей части
    initial = [points[0],points[len(points)/3+1],points[len(points)*2/3+1]]

    clusters = [Cluster([p],distanceFunc) for p in initial]
    Iter = 0
    # На всякий случай ограничиваем количество итераций 15. Можно, наверное, это ограничение убрать в будущем
    while Iter<15:
    	lists = [ [] for c in clusters]

    	# для каждой точки определяем ближайший центройд и таким образом относим ее к кластеру
        for p in points:
            smallest_distance = distanceFunc(p,clusters[0].centroid)
            index = 0
            for i in range(len(clusters[1:])):
                distance = distanceFunc(p, clusters[i+1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i+1
            lists[index].append(p)

        biggest_shift = 0.0
        # расчет максимального смещения (biggest_shift) центройдов по отношению к предыдущей итерации
        numCl = len(clusters)
        for i in range(numCl):
			if len(lists[i])==0:
				del clusters[i]

        for i in range(len(clusters)):
            shift = clusters[i].update(lists[i],distanceFunc)
            biggest_shift = max(biggest_shift, shift)

        if biggest_shift < cutoff:
            break
        Iter = Iter + 1
    return clusters

def UkkonenDistance(a, b):
	# Расстояние Юкконена.
	# Ссылка на статью:
	# Этот метод, в действительности, обычная косинусная мера двух векторов (Codine Distance)
	# однако используется только в привязке к методу Юкконена, поэтому и такое название
	# Само преобразование цепочки в вектор - это отдельный метод 

    d = 0
    A2 = 0
    B2 = 0
    for i in range(a.dim):
    	d = d + a.coords[i]*b.coords[i]
        A2 = A2 + a.coords[i]*a.coords[i]
        B2 = B2 + b.coords[i]*b.coords[i]
    d = 1 - d/(math.sqrt(A2) + math.sqrt(B2))
    return d

def RandomDistance(a, b):
	# Случайное расстояние. Используется исключительно как Baseline. Точка отсчета для остальных методов.
    return random.random()

def LevenshteinDistance(a, b):
	# Расстояние Левенштейна 
	seq1 = a.elements
	seq2 = b.elements
	oneago = None
	thisrow = range(1, len(seq2) + 1) + [0]
	for x in xrange(len(seq1)):
		twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
		for y in xrange(len(seq2)):
			delcost = oneago[y] + 1
			addcost = thisrow[y - 1] + 1
			subcost = oneago[y - 1] + (seq1[x] != seq2[y])
			thisrow[y] = min(delcost, addcost, subcost)
	return thisrow[len(seq2) - 1]

def EditDistance(a, b):
	# Редакторское расстояние.
	# Обычно редакторское расстояние приравнивают к расстоянию Левенштейна и даже говорят о том, что это одно и тоже
	# Однако, это не совсем так: в редакторском расстонии вес замены символа равен 2 (см. переменную subcost), а не 1
	seq1 = a.elements
	seq2 = b.elements
	oneago = None
	thisrow = range(1, len(seq2) + 1) + [0]
	for x in xrange(len(seq1)):
		twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
		for y in xrange(len(seq2)):
			delcost = oneago[y] + 1
			addcost = thisrow[y - 1] + 1
			subcost = oneago[y - 1] + 2*(seq1[x] != seq2[y])
			thisrow[y] = min(delcost, addcost, subcost)
	return thisrow[len(seq2) - 1]

def KendallDistance(a, b):
	# Классическая статиситческая мера Кендалла
 	# http://en.wikipedia.org/wiki/Kendall_tau_distance
 	seq1 = a.elements
	seq2 = b.elements
	fullOrder = {}

	for i in seq1:
		if i in seq2:
			fullOrder[i]= (seq1.index(i),seq2.index(i))

	count = 0

	for i in fullOrder:
		for j in fullOrder:
			if ((fullOrder[i][1]-fullOrder[j][1]) * (fullOrder[i][0]-fullOrder[j][0]) < 0) and (i>j):
				count = count + 1

	if len(fullOrder)<2:
		return 	random.gauss(0.5, 0.2)

	return 2.0*count/len(fullOrder)/(len(fullOrder)-1)

def SetOrders(orders, numOfChains, numOfSets, avgLength, maxLength, disp = 0):
	# по входящим полностью упорядоченным множествам мы формируем набор цепочек (частично упорядоченных множеств)
	# orders - набор полностью упорядоченных множеств
	# numOfChains - количество цепочек, которое будет создано на основе одного orders
	# numOfSets - пока не трогаем.
	# avgLength, maxLengthm - средняя и максимальная длина цепочек. Нужно для случая генерации цепочек разных длин
	# disp - дисперсия распределения длин цепочек. Нужно для случая генерации цепочек разных длин

	chains = []

	for i in range(numOfSets):
		for j in range(len(orders)):
			temp = GenerateChains(numOfChains,orders[j], avgLength, maxLength, disp)
			for k in range(numOfChains):
				chains.append(Chain(temp[k],j*numOfChains+k))
	#len(orders) по сути количество полностью упорядоченных множеств и является количеством кластеров в данных
	#chains - сформированные цепочеки
	#len(orders[0]) - количество элементов в полностью упорядоченном множестве. мы считаем, что для всех orders оно одинаково в рамках эксперимента
	return len(orders) , chains, len(orders[0])

def GenerateChains(numOfChains, order, avgLength, maxLength, disp = 0 ):
	# создает цепочку по входящему упорядоченному множество и соответвующей длины и дисперсии

	elements = []
	for i in range(numOfChains):

		# если средняя длина равна максимальной, то мы считаем, что нам все цепочки нужны одной длины
		if avgLength == maxLength:
			length = avgLength
		else:
			# если средняя длина не равна максимальной, то мы считаем цепочки случайной длины с некоторым разбросом
			length = int(max(min(random.gauss(avgLength,disp), maxLength ),4))

		b = random.sample(order, length)
		b.sort()

		elements.append([])
		for j in range(length):
			elements[i].append(order[b[j]-1])
	return elements

def MappingToPoints(chains, setOrdersSize):
	# Метод перевода цепочек в обычное n-мерное пространоство в соответствии со статьей Юкконена

	points = []
	for i in range(len(chains)):
		summV = 0
		length = len(chains[i].elements) #m in formula (10)
		pointCoord = []
		for k in range(setOrdersSize):
			if k in chains[i].elements:
				pointCoord.append(-(length + 1)/2. + chains[i].elements.index(k) + 1.0)
			else:
				pointCoord.append(0)
			summV = summV  + (pointCoord[k])*(pointCoord[k])
		for k in range(setOrdersSize):
			pointCoord[k] = pointCoord[k] / math.sqrt(summV)
		points.append(Point(pointCoord,i))
	return points

def AdjustedRandIndex(clusters, numOfPointsInGroup):
	# Представим себе, что у нас есть результат кластеризации объектов
	# И есть правильное их разбиение. Нужно оценить качество кластеризации
	# Для этого нам нужен Adjusted Rand Index, реализованный в данном методе
	# Используется как оценка качества кластеризации и, как следствие, оценка качества функции расстояния
	# http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

	matrix = {}
	if len(clusters) < 3:
		clusters.append(Cluster([], LevenshteinDistance)) 

	if len(clusters) < 3:
		clusters.append(Cluster([], LevenshteinDistance)) 

	for i,c in enumerate(clusters):
		matrix[i] = {}
		for x in range(len(clusters)):
			matrix[i][x] = 0

		for p in c.points:
			matrix[i][p.idP/numOfPointsInGroup] = matrix[i][p.idP/numOfPointsInGroup] + 1

	summEl = 0

	for cluster in matrix:
		for cls in matrix[cluster]:
			if matrix[cluster][cls]>1:
				summEl = summEl + C_n_k(matrix[cluster][cls],2)

	n2 = C_n_k(numOfPointsInGroup*len(clusters),2)

	nj2 = 0
	ni2 = 0

	for i,c in enumerate(clusters):
		if len(c.points)>1:
			nj2 = nj2 + C_n_k(len(c.points),2)
		else:
			nj2 = nj2 + 1

	ni2 = C_n_k(numOfPointsInGroup,2)*len(clusters)

	return (summEl - ni2*nj2/n2) / (0.5 * (ni2+nj2) - ni2*nj2/n2)

def PrintClusteringResult(clusters,numOfPointsInGroup):
	# Признаться, не помню, работает ли
	for i,c in enumerate(clusters):
		for p in c.points:
			print i,p,p.idP/numOfPointsInGroup

def C_n_k (n,k):
	# Вспомогательная функция. Расчет классического комбинаторного сочетания из n по k.
	return math.factorial(n)/math.factorial(n-k)/math.factorial(k)

def FirstTest(orders, NumberOfTests = 100, numOfPointsInEachGroup = 30, numOfSets = 1):
	# we need to test different measures depending on:
	# 	1) size of full order (n = 20, 30 or 40)
	# 	2) length of chains (L = 5..n)
	# number of clusters is k=3 and number of clusterization is NumberOfTests = 100 by default

	# important thing: we need to get not average, but full set of results:
	# 3 [n] * |L| * 100 [NumberOfTest] for each distance

	print "----------------FirstTest----------------------"

	Ukk, Rand, Kend, Leven, Edit = {},{},{},{},{}

	for i in orders:
		Ukk[i], Rand[i], Kend[i], Leven[i], Edit[i] = {},{},{},{},{}

		for L in range(5,i-1):
		#may be not good, but work. i - index in dict of orders and in same time number of element in order. From i/2 to i-1

			Edit[i][L], Ukk[i][L], Rand[i][L], Kend[i][L], Leven[i][L] = [],[],[],[],[]

			for j in range(NumberOfTests):

				realClusterNum, chains, setOrdersSize = SetOrders(orders[i],numOfPointsInEachGroup, numOfSets, L, L)
				
				points = MappingToPoints(chains, setOrdersSize)
				k, cutoff = realClusterNum, 0.1

				clustersUkkonen = kmeans(points, k, cutoff, UkkonenDistance)
				Ukk[i][L].append(AdjustedRandIndex(clustersUkkonen,numOfPointsInEachGroup))

				clustersLevenshtein = kmeans(chains, k, cutoff, LevenshteinDistance)
				Leven[i][L].append(AdjustedRandIndex(clustersLevenshtein,numOfPointsInEachGroup))

				clustersKendall = kmeans(chains, k, cutoff, KendallDistance)
				Kend[i][L].append(AdjustedRandIndex(clustersKendall,numOfPointsInEachGroup))

				clustersEdit = kmeans(chains, k, cutoff, EditDistance)
				Edit[i][L].append(AdjustedRandIndex(clustersEdit,numOfPointsInEachGroup))

				clustersRand = kmeans(chains, k, cutoff, RandomDistance)
				Rand[i][L].append(AdjustedRandIndex(clustersRand,numOfPointsInEachGroup))

				print i,L,j,'Ukkonen',Ukk[i][L][j]
				print i,L,j,'Levenshtein',Leven[i][L][j]
				print i,L,j,'Kendall',Kend[i][L][j]
				print i,L,j,'Edit',Edit[i][L][j]
				print i,L,j,'Random',Rand[i][L][j]

	print "--------------------------------------------------------"
	return 0

def SecondTest(orders, NumberOfTests = 100, numOfPointsInEachGroup = 30, numOfSets = 1):
	# we need to test different measures depending on size of full order (n) with not stable L (L<=n)
	# number of clusters is k=3 and number of clusterization is NumberOfTests = 100

	# important thing: we need to get not average, but full set of results:
	# 3 [n] * |L| * 100 [NumberOfTest] for each distance

	print "----------------SecondTest----------------------"

	Ukk, Rand, Kend, Leven, Edit = {},{},{},{},{}

	for i in orders:
		Ukk[i], Rand[i], Kend[i], Leven[i], Edit[i] = {},{},{},{},{}

		for disp in range(12):
			d = (disp+1)*0.5

			Ukk[i][disp], Rand[i][disp], Kend[i][disp], Leven[i][disp], Edit[i][disp] = [],[],[],[],[]

			for j in range(NumberOfTests):

				realClusterNum, chains, setOrdersSize = SetOrders(orders[i],numOfPointsInEachGroup, numOfSets, int(len(orders[i][0])*2/3), len(orders[i][0]), d)
				points = MappingToPoints(chains, setOrdersSize)
				k, cutoff = realClusterNum, 0.1

				clustersUkkonen = kmeans(points, k, cutoff, UkkonenDistance)
				Ukk[i][disp].append(AdjustedRandIndex(clustersUkkonen,numOfPointsInEachGroup))

				clustersLevenshtein = kmeans(chains, k, cutoff, LevenshteinDistance)
				Leven[i][disp].append(AdjustedRandIndex(clustersLevenshtein,numOfPointsInEachGroup))

				clustersKendall = kmeans(chains, k, cutoff, KendallDistance)
				Kend[i][disp].append(AdjustedRandIndex(clustersKendall,numOfPointsInEachGroup))

				clustersEdit = kmeans(chains, k, cutoff, EditDistance)
				Edit[i][disp].append(AdjustedRandIndex(clustersEdit,numOfPointsInEachGroup))

				clustersRand = kmeans(chains, k, cutoff, RandomDistance)
				Rand[i][disp].append(AdjustedRandIndex(clustersRand,numOfPointsInEachGroup))

				print i,j,int(len(orders[i][0])*2/3),d,"Ukkonen",Ukk[i][disp][j]
				print i,j,int(len(orders[i][0])*2/3),d,"Levenshtein",Leven[i][disp][j]
				print i,j,int(len(orders[i][0])*2/3),d,"Kendall",Kend[i][disp][j]
				print i,j,int(len(orders[i][0])*2/3),d,"Edit",Edit[i][disp][j]
				print i,j,int(len(orders[i][0])*2/3),d,"Rand",Rand[i][disp][j]
				
	print "--------------------------------------------------------"
	return 0

def main():

	#  Запуск основной процедуры. Базовые переменные:
	#  orders - набор базовых полностью упорядоченных множеств, на основе которых будут формироваться частично упорядоченные множества
	#  orders могут содержать различное количество элементов и заданы просто для тестирования из головы
	#  orders должны отличаться друг от друга порядком следования элементов
	#  
	#  numOfPointsInEachGroup. Эта переменная показывает количество цепочек (частично упорядоченных множеств), которые будут сформированы из каждого orders
	#  numOfSets = 1. Переменная позовляет варьировать количество групп таких цепочек из каждого orders. Всегда равен 1. Пока не имеет смысл трогать.
	#  NumberOfTests. По сути показывает сколько раз будет произведен каждый эксперимент.

	orders = {}

	# orders[20] = [] # 20 elements in order
	# orders[20].append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
	# orders[20].append([19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
	# orders[20].append([3,2,19,1,9,6,10,18,17,14,15,16,11,12,8,5,4,7,13,0])

	# orders[30] = [] # 30 elements in order
	# orders[30].append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
	# orders[30].append([29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
	# orders[30].append([21,3,2,19,22,20,23,24,1,9,6,10,18,17,28,29,14,15,16,11,27,25,26,12,8,5,4,7,13,0])

	orders[40] = [] # 40 elements in order
	orders[40].append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
	orders[40].append([39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
	orders[40].append([39,21,3,2,19,22,20,23,24,1,9,6,10,18,17,28,29,14,35,34,33,32,31,30,15,16,11,27,25,26,12,8,5,4,7,13,0,38,37,36])

	FirstTest(orders, NumberOfTests=1, numOfPointsInEachGroup = 30, numOfSets = 1)
	# SecondTest(orders, NumberOfTests=1, numOfPointsInEachGroup = 30, numOfSets = 1)
	
if __name__ == "__main__":
    main()
