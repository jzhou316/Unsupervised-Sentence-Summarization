import os


def copy_rate(sent1, sent2):
    """
    copy_rate between two sentences.

    Input:
        sent1, sent2: two sentence strings (generated summary, source).
    Output:
        score: copy rate on unigrams.
    """

    sent1_split = set(sent1.split())
    sent2_split = set(sent2.split())
    intersection = sent1_split.intersection(sent2_split)
#     recall = len(intersection) / len(sent2_split)
    precision = len(intersection) / len(sent1_split)
#     union = sent1_split.union(sent2_split)
#     jacd = 1 - len(intersection) / len(union)
#     score = stats.hmean([recall, precision])
#     score = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0

    return precision


sourcepath = '/n/rush_lab/users/jzhou/LM/data/Giga-sum/input_unk.txt'
refpath = '/n/rush_lab/users/jzhou/LM/data/Giga-sum/task1_ref0.txt'

'''
gensmrypath = '/n/rush_lab/users/jzhou/5.0_cluster/results_untied/smry_input_unk_Ks10_clust0_temper10.0_ELcat_eosavg0_n6_ns10_nf300_a0.1_b0.0_all.txt'
results_dir = './results_untied/'
'''

gensmrypath = './results_gpt2/smry_input_unk_Ks10_clust1_n6_ns10_nf300_a0.1_b0.0_all.txt'
results_dir = './results_gpt2/'

lp = 0.1
# lp = 0.2


g = open(sourcepath, 'r')
arts = [line.strip().strip(' .') for line in g if line.strip()]
g.close()

g = open(refpath, 'r')
refs = [line.strip() for line in g if line.strip()]
g.close()

g = open(gensmrypath, 'r')
lines = [line.strip() for line in g if line.strip()]
g.close()

b = 1.0
# gensmrypath_new = f'/media/work/LM/4.0_cluster/results/GenSmry_3eos_input_a0.1_b{b}_single.txt'

basename = os.path.basename(gensmrypath)
basename = os.path.splitext(basename)[0]
    
gensmrypath_new = results_dir + basename.replace('b0.0_all', f'b{b}_single') + '.txt'

g = open(gensmrypath_new, 'w')

i = 0
j = 1
count = 0
cp_rate = []
lens = []
comp_rate = []
while j <= len(lines):
    if j == len(lines) or lines[j].startswith('-----') and not lines[j].startswith('----- '):
        count += 1
        # from i to j-1
        curl = lines[(i + 1):j]
        ssa = [(curl[k], curl[k + 1], curl[k + 2]) for k in range(len(curl)) if k % 3 == 0]
        ssa = sorted(ssa, key = lambda x: float(x[1].split()[0]) / (len(x[0].split()) - lp) ** b, reverse=True)
#         ssa = sorted(ssa, key = lambda x: float(x[1].split()[0]) / (len(x[0].split()) - 0.1) ** b, reverse=True)
#         ssa = sorted(ssa, key = lambda x: float(x[1].split()[2]) / (len(x[0].split()) - 0.5), reverse=True)

#         if arts[count - 1] == '<unk>':
#             g.write('\n')
#         else:
#             if len(ssa[0][0].split()) <= 1:
#                 g.write(arts[count - 1])
#                 g.write('\n')
#                 cp_rate.append(1)
#             else:
#                 g.write(' '.join(ssa[0][0].split()[:-1]))
#                 g.write('\n')
#                 cp_rate.append(copy_rate(' '.join(ssa[0][0].split()[:-1]), arts[count - 1]))

        if len(ssa[0][0].split()) <= 1:
            g.write(arts[count - 1])
            g.write('\n')
            cp_rate.append(1)
            comp_rate.append(1)
            lens.append(len(arts[count - 1].split()))
        else:
            g.write(' '.join(ssa[0][0].split()[:-1]))
            g.write('\n')
            cp_rate.append(copy_rate(' '.join(ssa[0][0].split()[:-1]), arts[count - 1]))
            comp_rate.append(len(ssa[0][0].split()[:-1]) / len(arts[count - 1].split()))
            lens.append(len(ssa[0][0].split()[:-1]))
        i = j    
    j += 1

g.close()

os.system('sed -i "s/<unk>/UNK/g" ' + gensmrypath_new)
print('copy rate: %f' % (sum(cp_rate) / len(cp_rate)))
print('compression rate: %f' % (sum(comp_rate) / len(comp_rate)))
print('average summary length: %f' % (sum(lens) / len(lens)))
os.system('files2rouge ' + gensmrypath_new + ' ' + refpath)


