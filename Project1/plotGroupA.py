# plotting of group A

import matplotlib.pyplot as plt
import csv

with open('Project1_data/groupA.csv', 'rb') as csvfile:
	data = list(csv.reader(csvfile))

red = ("red")
blue = ("blue")
maleX = []
maleY = []
femaleX = []
femaleY = []
tp = 0
tn = 0
fp = 0
fn = 0


for el in data:
	if int(el[2]) == 0: # if it's a male data point
		maleX.append(el[0])
		maleY.append(el[1])

		# test whether prediction was correct
		distance = ( ((float(el[1]) - 160) * (6.5 - 5.45)) - ((float(el[0]) - 5) * (100 - 165)) )

		if distance > 0:
			tn = tn + 1
		else:
			fn = fn + 1


	elif int(el[2]) == 1: # if it's a female data point'
		femaleX.append(el[0])
		femaleY.append(el[1])
		# test whether prediction was correct
		distance = ( ((float(el[1]) - 160) * (6.5 - 5.45)) - ((float(el[0]) - 5) * (100 - 165)) )
		if distance <= 0:
			tp = tp + 1
		else:
			fp = fp + 1


print('tp=' + str(tp))
print('fp=' + str(fp))
print('tn=' + str(tn))
print('fn=' + str(fn))

# plotting the male points
plt.scatter(maleX, maleY, c=red, label = "Males")

# plotting the female points
plt.scatter(femaleX, femaleY, c=blue, label = "Females")

# draw linear separator
plt.plot([5, 6.5], [165, 100], color='g', linewidth=2.0);

# plot arrow to show decision (but need to find perpendicular line/slope inverse and midpoint)
# plt.arrow(5.95, 150, 1, 1, color='g', linewidth=2.0);

# giving a title to my graph
plt.title('Group A')
plt.xlabel('Height (in feet)')
plt.ylabel('Weight (in pounds)')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
