from windrose import WindroseAxes
import matplotlib.pyplot as plt
import numpy as np

ws1 = np.random.random(500) * 6
wd1 = np.random.random(500) * 360
ws2 = np.random.random(500) * 6
wd2 = np.random.random(500) * 360
ws3 = np.random.random(500) * 6
wd3 = np.random.random(500) * 360
ws4 = np.random.random(500) * 6
wd4 = np.random.random(500) * 360

fig=plt.figure(figsize=(15,15),dpi=150)
# rect = [lowerleft_x,lowerleft_y,width,height]

opening=0.99


rect1=[0.1, 0.5, 0.4, 0.4] 
wa1=WindroseAxes(fig, rect1)
fig.add_axes(wa1)
wa1.set_title('00:00 - 06:00')
wa1.bar(wd1, ws1, normed=True, opening=opening, edgecolor='white')

rect2=[0.6, 0.5, 0.4, 0.4]
wa2=WindroseAxes(fig, rect2)
fig.add_axes(wa2)
wa2.set_title('06:00 - 12:00')
wa2.bar(wd2, ws2, normed=True, opening=opening, edgecolor='white')

rect3=[0.1,0,0.4,0.4] 
wa3=WindroseAxes(fig, rect3)
fig.add_axes(wa3)
wa3.set_title('12:00 - 18:00')
wa3.bar(wd3, ws3, normed=True, opening=opening, edgecolor='white')

rect4=[0.6,0,0.4,0.4] 
wa4=WindroseAxes(fig, rect4)
fig.add_axes(wa4)
wa4.set_title('18:00 - 00:00')
wa4.bar(wd4, ws4, normed=True, opening=opening, edgecolor='white')
wa4.set_legend()

plt.show()


