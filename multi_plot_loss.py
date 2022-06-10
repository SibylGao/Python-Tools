from cProfile import label
import json
from turtle import color
import matplotlib.pyplot as plt

def read_loss(filepath,base_loss=False,feature_loss=False):
    epoch = []
    train_loss = []
    test_acc1 = []
    with open(filepath) as f:
        for line in f.readlines():
            tmp_dict = json.loads(line)
            epoch_tmp = tmp_dict['epoch']
            epoch.append(epoch_tmp)

            if base_loss ==False:
                train_loss_tmp = tmp_dict['train_loss']
                train_loss.append(train_loss_tmp)
            elif feature_loss==True:
                train_loss_tmp = tmp_dict['train_feature_loss']
                train_loss.append(train_loss_tmp)
            else:
                train_loss_tmp = tmp_dict['train_base_loss']
                train_loss.append(train_loss_tmp)
                
            test_acc1_tmp = tmp_dict['test_acc1']
            test_acc1.append(test_acc1_tmp)
    return epoch, train_loss, test_acc1

# save_path = 

def read_distillation_loss(filepath):
    epoch = []
    train_loss = []
    test_acc1 = []
    train_feature_loss = []
    train_base_loss = []
    train_distillation_loss = []
    with open(filepath) as f:
        for line in f.readlines():
            tmp_dict = json.loads(line)
            epoch_tmp = tmp_dict['epoch']
            epoch.append(epoch_tmp)

            train_loss_tmp = tmp_dict['train_loss']
            train_loss.append(train_loss_tmp)
        
            train_feature_loss_tmp = tmp_dict['train_feature_loss']
            train_feature_loss.append(train_feature_loss_tmp)
        
            train_base_loss_tmp = tmp_dict['train_base_loss']
            train_base_loss.append(train_base_loss_tmp)
            
            train_distillation_loss_tmp = tmp_dict['train_distillation_loss']
            train_distillation_loss.append(train_distillation_loss_tmp)

            test_acc1_tmp = tmp_dict['test_acc1']
            test_acc1.append(test_acc1_tmp)

    return epoch, train_loss, test_acc1,train_base_loss,train_feature_loss,train_distillation_loss


file_path1 = "/Users/gaoshiyu01/Desktop/experiment/deit_no_distillation/log.txt"
file_path2 = "/Users/gaoshiyu01/Desktop/experiment/deit_inter_log_weight0dot1/log.txt"
file_path3 = "/Users/gaoshiyu01/Desktop/experiment/deit_stu_asc/log.txt"
file_path4 = "/Users/gaoshiyu01/Desktop/experiment/inter_non_weightloss/log.txt"


epoch_1, train_loss1, test_acc1_1 = read_loss(file_path1)
epoch_2, train_loss2, test_acc1_2 = read_loss(file_path2)
epoch_3, train_loss3, test_acc1_3 = read_loss(file_path3)
epoch_4, train_loss4, test_acc1_4 = read_loss(file_path4)

# plt.figure()
# plt.plot(epoch_1,train_loss1,color='b',label='no distillation')
# plt.plot(epoch_2,train_loss2,color='g',label='Inter weight 0.1')
# plt.plot(epoch_3,train_loss3,color='r',label='Deit stu asc')
# plt.plot(epoch_4,train_loss4,color='y',label='Inter non weightloss')
# plt.title("feature loss")
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('train loss')
# plt.savefig('feature_loss.png')


# plt.figure()
# plt.plot(epoch_1,test_acc1_1,color='b',label='no_distillation')
# plt.plot(epoch_2,test_acc1_2,color='g',label='Inter weight 0.1')
# plt.plot(epoch_3,test_acc1_3,color='r',label='Deit stu asc')
# plt.plot(epoch_4,test_acc1_4,color='y',label='Inter non weightloss')
# plt.legend(loc='lower right')
# plt.xlabel('epoch')
# plt.ylabel('test acc1')
# plt.savefig('test_acc1.png')


epoch_2, train_loss2, test_acc1_2, train_base_loss2, train_feature_loss2, train_distillation_loss2 = read_distillation_loss(file_path2)
epoch_3, train_loss3, test_acc1_3, train_base_loss3, train_feature_loss3, train_distillation_loss3 = read_distillation_loss(file_path3)
epoch_4, train_loss4, test_acc1_4, train_base_loss4, train_feature_loss4, train_distillation_loss4 = read_distillation_loss(file_path4)
plt.figure()
plt.plot(epoch_2,train_feature_loss2,color='g',label='Inter weight 0.1 hidden')
plt.plot(epoch_3,train_feature_loss3,color='r',label='Student asc hidden')
plt.plot(epoch_4,train_feature_loss4,color='y',label='Inter non weight hidden')

plt.plot(epoch_2,train_distillation_loss2,color='g',label='Inter weight 0.1 logit',linestyle="--")
plt.plot(epoch_3,train_distillation_loss3,color='r',label='Student asc logit',linestyle="--")
plt.plot(epoch_4,train_distillation_loss4,color='y',label='Inter non weight logit',linestyle="--")
plt.title("losses between teacher and student")
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('feature_loss.png')