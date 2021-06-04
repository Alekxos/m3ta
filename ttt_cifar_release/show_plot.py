import numpy as np
import torch
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('colorblind')

results_dir = "/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/"

corruptions_names = ['original', 'gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
						'snow', 'frost', 'fog', 'bright', 'contrast', 'elastic', 'pixelate', 'jpeg']

corruptions_names_short = ['orig', 'gauss', 'shot', 'impul', 'defoc', 'glass', 'motn', 'zoom', 
						'snow', 'frost', 'fog', 'brit', 'contr', 'elast', 'pixel', 'jpeg']
info = []
info.append(('gn', '_expand', 5))
# info.append(('gn', '_expand_final', 5))
# info.append(('gn', '_expand_final', 4))
# info.append(('gn', '_expand_final', 3))
# info.append(('gn', '_expand_final', 2))
# info.append(('gn', '_expand_final', 1))
# info.append(('bn', '_expand_final', 5))

########################################################################

def easy_barplot(table, fname, width=0.2):
	labels = ['Test-time training online', 'MeTTTa'] # ['Baseline', 'Joint training', 'Test-time training', 'Test-time training online']
	index =  np.asarray(range(len(table[0,:])))

	plt.figure(figsize=(9, 2.5))
	for i, row in enumerate(table):
		plt.bar(index + i*width, row, width, label=labels[i])

	plt.ylabel('Error (%)')
	plt.xticks(index + width/4, corruptions_names)
	plt.xticks(rotation=45)
	plt.legend(prop={'size': 8})
	plt.tight_layout(pad=0)
	plt.savefig(fname)
	plt.close()

def easy_latex(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '& %.1f' %(entry)
			else:
				row_str += '& %s' %(entry)
		print(row_str)

# for parta, partb, level in info:
# 	print(level, parta + partb)
# 	print(f"PATH: {results_dir}C10C_layer2_{level}_{parta}{partb}.pth")
# 	results = torch.load(f'{results_dir}C10C_layer2_{level}_{parta}{partb}.pth')
# 	if parta == 'bn':
# 		results = results[0:3,:]
#
# 	easy_barplot(results, f'{results_dir}C10C_layer2_{level}_{parta}{partb}.pdf')
# 	easy_latex([corruptions_names_short], prec1=False)
# 	easy_latex(results)

results = torch.load('/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_mettta/C10C_layer2_5_gn_expand_final.pth')
easy_barplot(results, '/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_mettta/C10C_layer2_5_gn_expand_final.pdf')
easy_latex([corruptions_names_short], prec1=False)
easy_latex(results)