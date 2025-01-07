import argparse
import importlib
import torch.nn
import torch.nn.functional as F
import provider
import os
import sys
# from model_pytorch import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # PlantNet_pytorch/models
ROOT_DIR = os.path.dirname(BASE_DIR)                # PlantNet_pytorch
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.test_utils import *
from utils.clustering import cluster

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='model_pytorch', help='model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--verbose', default=True, help='if specified, output color-coded seg obj files')
    parser.add_argument('--log_dir', default='out', help='Log dir [default: log]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
    parser.add_argument('--bandwidth', type=float, default=0.6, help='Bandwidth for meanshift clustering [default: 1.]')
    parser.add_argument('--input_list', type=str, default='data/test_file_list.txt',
                        help='Input data list file')
    parser.add_argument('--model_path', type=str,
                        default=r'',
                        help='Path of model')

    return parser.parse_args()
args = parse_args()

NUM_CLASSES = 2#语义分类标签


BATCH_SIZE = 1
NUM_POINT = args.num_point
GPU_INDEX = args.gpu
MODEL_PATH = args.model_path
TEST_FILE_LIST = args.input_list
BANDWIDTH = args.bandwidth
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
mean_num_pts_in_group = np.loadtxt("PlantNet/PlantNet_pytorch/log/mean_ins_size.txt")#loadtxt(os.path.join(MODEL_PATH.split('/')[5], '../mean_ins_size.txt'))

output_verbose = args.verbose  # If true, output all color-coded segmentation obj files

LOG_DIR = os.path.join(ROOT_DIR, 'log')  # PlantNet_pytorch/log
OUTPUT_DIR = os.path.join(LOG_DIR, args.log_dir)  # PlantNet_pytorch/log/out
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

# OUTPUT_DIR = os.path.join(LOG_DIR, 'log_1') # PlantNet_pytorch/log/out/log_1
# if not os.path.exists(OUTPUT_DIR):
#     os.mkdir(OUTPUT_DIR)


LOG_FOUT = open(os.path.join(OUTPUT_DIR, 'log_inference.txt'), 'w') # PlantNet_pytorch/log/out/log_inference.txt
LOG_FOUT.write(str(args)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def test():
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.plantnet_model(NUM_CLASSES).to(device)

    File_PATH_LIST = [os.path.join(ROOT_DIR, '/'.join(line.rstrip().split('/')[2:])) for line in open(os.path.join(ROOT_DIR, args.input_list))] # ['PlantNet_pytorch/data/FPSprocessed/test_h5/test.h5']
    len_pts_files = len(File_PATH_LIST) # 1
    with torch.no_grad():
        # 加载模型权重参数
        checkpoint_dir = os.path.join(LOG_DIR, 'checkpoints')
        try:
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string("Model restored.")
            classifier = classifier.eval()
        except:
            log_string("Model failed to restore.")
            return
        # checkpoint = torch.load(MODEL_PATH)
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        # log_string("Model restored.")
        # classifier = classifier.eval()

        output_filelist_f = os.path.join(OUTPUT_DIR, 'output_filelist1.txt')    # PlantNet_pytorch/log/out/output_filelist1.txt
        fout_out_filelist = open(output_filelist_f, 'w')
        for shape_idx in range(len_pts_files):
            file_path = File_PATH_LIST[shape_idx]   # 'PlantNet_pytorch/data/FPSprocessed/test_h5/test.h5'
            log_string('%d / %d ...' % (shape_idx+1, len_pts_files))
            log_string('Loading test file: ' + file_path)  
            out_data_label_filename = os.path.dirname(file_path).split('/')[-1] + '_pred.txt'   # 'test_h5_pred.txt'
            out_data_label_filepath = os.path.join(OUTPUT_DIR, out_data_label_filename)  # PlantNet_pytorch/log/out/test_h5_pred.txt
            out_gt_label_filename = os.path.dirname(file_path).split('/')[-1] + '_gt.txt'  # 'test_h5_gt.txt'
            out_gt_label_filepath = os.path.join(OUTPUT_DIR, out_gt_label_filename) # PlantNet_pytorch/log/out/test_h5_gt.txt
            fout_data_label = open(out_data_label_filepath, 'w')
            fout_gt_label = open(out_gt_label_filepath, 'w')

            fout_out_filelist.write(out_data_label_filepath+'\n')

            cur_data, cur_feature, cur_group, cur_sem, cur_obj = provider.load_h5_data_label_seg(file_path)
            cur_data = cur_data[:, :, :]    # (20, 4096, 3)
            cur_feature = cur_feature[:, :, :]  # (20, 4096, 3)
            cur_sem = np.squeeze(cur_sem)   # (20, 4096)
            cur_group = np.squeeze(cur_group)   # (20, 4096)
            # cur_obj = np.squeeze(cur_obj)
            

            cur_pred_sem = np.zeros_like(cur_sem)   # (20, 4096)
            cur_pred_sem_softmax = np.zeros([cur_sem.shape[0], cur_sem.shape[1], NUM_CLASSES])  # (20, 4096, 2)
            group_output = np.zeros_like(cur_group) # (20, 4096)
            # group_obj = np.zeros_like(cur_obj)
            
            
            num_data = cur_data.shape[0]    # 20, 因为batch_size=1，所以也是batch数量
            for j in range(num_data):   # 20
                log_string("Processsing: File [%d] (Batch[%d]/Batch[%d])"%(shape_idx+1, j+1, num_data+1))

                pts = cur_data[j,...]   # (4096, 3)
                # obj = cur_obj[j,...]    
                pointclouds_pl = np.expand_dims(pts, 0).astype(np.float32)  # (1, 4096, 3)
                pointclouds_pl = torch.from_numpy(pointclouds_pl)
                pointclouds_pl = pointclouds_pl.to(torch.device('cuda'))
                pred_sem, pred_ins, fuse_catch = classifier(pointclouds_pl) #(1,4096,2), (1,4096,5), (1,4096,4096)
                pred_sem_softmax = F.softmax(pred_sem, dim=2)   # (1, 4096, 2)
                pred_sem_label = torch.argmax(pred_sem_softmax, dim=2)  # (1, 4096)
                pred_ins = np.squeeze(pred_ins, axis=0) # (4096, 5)
                pred_sem = np.squeeze(pred_sem_label, axis=0)   # (4096,)
                pred_sem_softmax = np.squeeze(pred_sem_softmax, axis=0) # (4096, 2)
                pred_sem = pred_sem.cpu().detach().numpy()
                pred_sem_softmax = pred_sem_softmax.cpu().detach().numpy()
                cur_pred_sem[j, :] = pred_sem
                cur_pred_sem_softmax[j, ...] = pred_sem_softmax# batch_size x data_number x num_class
                
                # cluster
                bandwidth = BANDWIDTH
                pred_ins = pred_ins.cpu().detach().numpy()
                num_clusters, labels, cluster_centers = cluster(pred_ins, bandwidth)
                #最终聚类的数量，每个点的标签，每个簇的中心
                groupids_block = labels # (4096,)每个点属于哪个簇的标签 (4096,)
                # group_obj[j,:] = obj
                
                un = np.unique(groupids_block)  #每个簇的标签
                pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
                group_pred_final = -1 * np.ones_like(groupids_block)    # (4096,)没分到簇的点标记为-1
                grouppred_cnt = 0
                for ig, g in enumerate(un): #each object in prediction
                    if g == -1:
                        continue
                    tmp = (groupids_block == g) # 属于g这个簇的点
                    sem_seg_g = int(stats.mode(pred_sem[tmp], keepdims=False)[0])   #簇g中出现最多的语义标签
                    if np.sum(tmp) > 0.01 * mean_num_pts_in_group[sem_seg_g]:   #如果簇g中的点数大于平均点数的1%
                        group_pred_final[tmp] = grouppred_cnt   #簇g中的点都标记为grouppred_cnt
                        pts_in_pred[sem_seg_g] += [tmp]  #添加簇g到语义标签为sem_seg_g的语义类别中
                        grouppred_cnt += 1  #到下一个簇
                
                group_output[j, :] = group_pred_final

            group_pred = group_output.reshape(-1)   # (20*4096,)
            seg_pred = cur_pred_sem.reshape(-1)  # (20*4096,)
            seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, NUM_CLASSES])  # (20*4096, 2)
            pts = cur_data.reshape([-1, 3]) # (20*4096, 3)
            feature = cur_feature.reshape([-1, 3])  # (20*4096, 3)
            # obj_gt = group_obj.reshape(-1)
            seg_gt = cur_sem.reshape(-1)    # (20*4096,)

            if output_verbose:
                ins = group_pred.astype(np.int32)
                sem = seg_pred.astype(np.int32)
                sem_softmax = seg_pred_softmax
                sem_gt = seg_gt
                ins_gt = cur_group.reshape(-1)  # (20*4096,)
                for i in range(pts.shape[0]):
                    fout_data_label.write('%f %f %f %d %d %d %f %d %d\n' % (
                    pts[i, 0], pts[i, 1], pts[i, 2], feature[i,0], feature[i,1], feature[i,2], sem_softmax[i, sem[i]], sem[i], ins[i]))   #预测txt的文件组成形式
                    fout_gt_label.write('%f %f %f %d %d %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], feature[i,0], feature[i,1], feature[i,2], sem_gt[i], ins_gt[i]))  #gt txt的文件组成形式

            fout_data_label.close()
            fout_gt_label.close()

        fout_out_filelist.close()

if __name__ == "__main__":
    test()
    LOG_FOUT.close()
