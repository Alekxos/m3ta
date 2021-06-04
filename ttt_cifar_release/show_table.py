from os import path
import sys
import numpy as np
import torch
from utils.misc import *
from test_calls.show_result import get_err_adapted

corruptions_names = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom',
					 'snow', 'frost', 'fog', 'bright', 'contra', 'elastic', 'pixel', 'jpeg']
corruptions_names.insert(0, 'orig')

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
corruptions.insert(0, 'original')

info, baseline = [], []
# info.append(('gn', '_expand_final', 5))
info.append(('bn', '_expand', 5))
# info.append(('gn', '_expand_final', 4))
# info.append(('gn', '_expand_final', 3))
# info.append(('gn', '_expand_final', 2))
# info.append(('gn', '_expand_final', 1))
# info.append(('bn', '_expand_final', 5))

for level in [5, ]:
	baseline += [('', '', level)]
	baseline += [('bn', '1_alp', level)]
	baseline += [('bn', '0.5_alp', level)]

########################################################################

def print_table(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '%.1f\t' %(entry)
			else:
				row_str += '%s\t' %(str(entry))
		print(row_str)

def show_table(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []
		try:
			rdict_ada = torch.load(path.join(folder, f'{corruption}_{level}_ada.pth'))
			inl_folder = '/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/C10C_layer2_online_gn_expand/'
			rdict_inl = torch.load(path.join(inl_folder, f'{corruption}_{level}_inl.pth'))

			ssh_confide = rdict_ada['ssh_confide']
			new_correct = rdict_ada['cls_correct']
			old_correct = rdict_inl['cls_correct']

			row.append(rdict_inl['cls_initial'])
			old_correct = old_correct[:len(new_correct)]
			err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
			row.append(err_adapted)

		except:
			row.append(-1)
			row.append(-1)
		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_none(folder, level):
	results = []
	for corruption in corruptions:
		try:
			rdict_inl = torch.load(path.join(folder, f'{corruption}_{level}_none.pth'))
			results.append(rdict_inl['cls_initial'])
		except:
			results.append(0)
	results = np.asarray([results])
	results = results * 100
	return results

# for parta, partb, level in info:
# 	print(level, parta + partb)
# 	print_table([corruptions_names], prec1=False)
# 	if parta == 'bn':
threshold = 0.9
	# else:
	# 	threshold = 1

	# results_none = show_none(f'{results_dir}C10C_none_none_{parta}', level)
	# print_table(results_none)
	#
	# print(f"none path: {results_dir}C10C_none_none_{parta}")

	# results_slow = show_table(f'{results_dir}C10C_layer2_slow_{parta}{partb}', level, threshold=threshold)
	# print_table(results_slow)

	# print(f"none path: {results_dir}C10C_layer2_slow_{parta}{partb}")
	#
online_dir = "/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/"
# print(f"online path: {online_dir}C10C_layer2_online_{parta}{partb}")
# results_onln = show_table(f'{online_dir}C10C_layer2_online_{parta}{partb}', level, threshold=threshold)
results_onln = show_table('/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/C10C_layer2_online_gn_expand', 5, threshold=1)
results_onln = results_onln[1:,:]
print("TTT  TABLE")
print_table(results_onln)

mettta_dir = "/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_mettta/"
# print(f"mettta path: {mettta_dir}C10C_layer2_mettta_{parta}{partb}")
# results_mettta = show_table(f'{mettta_dir}C10C_layer2_mettta_{parta}{partb}', level, threshold=0.9)
results_mettta = show_table('/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_mettta/C10C_layer2_online_bn_expand', 5, threshold=0.9)
results_mettta = results_mettta[1:, :]
print("METTTA TABLE")
print_table(results_mettta)

results = np.concatenate((results_onln, results_mettta)) # np.concatenate((results_none, results_slow, results_onln))
torch.save(results, f'{mettta_dir}C10C_layer2_5_gn_expand_final.pth')

# for parta, partb, level in baseline:
# 	if parta == '':
# 		print(level)
# 		print_table([corruptions_names], prec1=False)
# 		continue
# 	results_none = show_none(f'{results_dir}C10C_none_baseline_{parta}_bl_{partb}', level)
# 	print_table(results_none)
