import os
import logging

root_dir = './datasets/Rain200H'
# root_dir = './datasets/Rain200L'
# root_dir = './datasets/Rain1200'
# root_dir = './datasets/SPANet'
real_dir = './datasets/real'
log_dir = './logdir'
log_test_dir = './log_test/'
show_dir = './showdir'
model_dir = './models'
data_dir = os.path.join(root_dir, 'train/rain')
mat_files = os.listdir(data_dir)
num_datasets = len(mat_files)

aug_data = False

device_id = '0'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

