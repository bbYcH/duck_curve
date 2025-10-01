import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image

home = 150
coeffient = 1
#%% Prof.LI's method to clean temperature data
tem= np.load(r'Tem.npy')
tem[32,19]=(tem[32,18]+tem[32,20])/2
tem[326,24]=(tem[326,23]+tem[326,25])/2
#%%
tem_data = np.load(r'Tem_cleaned.npy')  # 清理后的摄氏度数据(原数据存在None值)
tem_data_fahrenheit = (tem_data * 9 / 5 + 32)  # 转换为华氏度
# tem_fah_max = np.nanmax(tem_data_fahrenheit)  # nan
# tem_fah_min = np.nanmin(tem_data_fahrenheit)  # nan
generation_data = np.load('generation.npy') * coeffient  # (day, user, hour)
# generation_max = np.max(generation_data)
# generation_min = np.min(generation_data)
load_data = np.load('load.npy') * coeffient  # (day, user, hour)
# load_max = np.max(load_data)
# load_min = np.min(load_data)
price_data = np.load('price.npy')  

grid_total_generation = np.sum(generation_data[:, 0:home, :], axis=1)
grid_total_load = np.sum(load_data[:, 0:home, :], axis=1)

# 创建TensorBoard writer
writer = SummaryWriter('runs/duck_curve_experiment')

duck_curve = grid_total_load - grid_total_generation
for day in range(365):
    oneday_duck = duck_curve[day, :]
    plt.figure(figsize=(10, 6))
    plt.plot(oneday_duck)
    plt.title(f'Duck Curve - Day {day + 1}')
    plt.xlabel('Hour')
    plt.ylabel('Load - Generation')
    
    # 将matplotlib图片保存到内存缓冲区
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # 将图片转换为PIL Image，然后转换为numpy数组
    img = Image.open(buf)
    img_array = np.array(img)
    
    # 将图片添加到TensorBoard (HWC格式转换为CHW格式)
    if len(img_array.shape) == 3:
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    
    writer.add_image(f'Duck_Curve/Day_{day + 1}', img_array, day)
    
    plt.close()  # 关闭图片以节省内存
    buf.close()

# 关闭writer
writer.close()

print('所有图片已保存到TensorBoard。运行 tensorboard --logdir=runs 来查看结果。')

# userid = 3
# L = np.load(r'load.npy')[:, userid - 1, :]
# G = np.load(r'generation.npy')[:, userid - 1, :]
#
# pass