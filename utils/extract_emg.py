from extract_from_pkl import extract_pkl_to_pd, pkl_to_pd
from emg_dataset_creator import emg_dataset

emg_dataset('action-net/ActionNet_train.pkl', 'train_val/D4_train.pkl')
emg_dataset('action-net/ActionNet_test.pkl', 'train_val/D4_test.pkl')

exit()

print(pkl_to_pd('action-net/action_net_dataset/S00_2.pkl').head(5))
print(pkl_to_pd('action-net/action_net_dataset/S00_2.pkl').columns)

exit()

extract_pkl_to_pd('action-net')

labels = pkl_to_pd('action-net/ActionNet_train.pkl')['labels'].to_list()
label_set = set(labels)
print(label_set)
print(len(label_set))