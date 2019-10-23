"""
Post-processing of the generated summary sentences:
1. From all the summary sentences based on beam search, select one based on some length penalty
2. Evaluation of Rouge scores, copy rate, compression rate, etc.
"""
import os
import argparse


def copy_rate(sent1, sent2):
    """
    copy rate between two sentences.
    In particular, the proportion of sentence 1 that are copied from sentence 2.

    Input:
        sent1, sent2: two sentence strings (generated summary, source).
    Output:
        score: copy rate on unigrams.
    """
    sent1_split = set(sent1.split())
    sent2_split = set(sent2.split())
    intersection = sent1_split.intersection(sent2_split)
    # recall = len(intersection) / len(sent2_split)
    precision = len(intersection) / len(sent1_split)
    # union = sent1_split.union(sent2_split)
    # jacd = 1 - len(intersection) / len(union)  # jacquard distance
    # score = stats.hmean([recall, precision])  # F1 score (need to import scipy.stats.hmean)
    # score = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0  # F1 score

    return precision


# =============== some default path arguments ==================================
src = '/n/rush_lab/users/jzhou/LM/data/Giga-sum/input_unk.txt'
ref = '/n/rush_lab/users/jzhou/LM/data/Giga-sum/task1_ref0.txt'

'''
gens = '/n/rush_lab/users/jzhou/5.0_cluster/results_untied/smry_input_unk_Ks10_clust0_temper10.0_ELcat_eosavg0_n6_ns10_nf300_a0.1_b0.0_all.txt'
save_dir = './results_untied/'
'''
gen = './results_gpt2/smry_input_unk_Ks10_clust1_n6_ns10_nf300_a0.1_b0.0_all.txt'
save_dir = './results_gpt2/'

lp = 0.1
# ===============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description='Post-processing and evaluation of the generated summary sentences')
    parser.add_argument('--src', type=str, default=src, help='source sentence path')
    parser.add_argument('--ref', type=str, default=ref, help='reference summary path')
    parser.add_argument('--gen', type=str, default=gen, help='generated summary path')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='directory to save the result')
    parser.add_argument('--lp', type=float, default=lp, help='length penalty (additive onto length)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # read in the source, reference, and generated summaries (a list of summaries for each source sentence)
    g = open(args.src, 'r')
    arts = [line.strip().strip(' .') for line in g if line.strip()]
    g.close()

    g = open(args.ref, 'r')
    refs = [line.strip() for line in g if line.strip()]
    g.close()

    g = open(args.gen, 'r')
    lines = [line.strip() for line in g if line.strip()]
    g.close()

    # length penalty for selecting the finished hypothesis from beam search
    # takes the form (length + lp) ^ b
    lp = args.lp
    b = 1.0

    # generate the new path to save the results
    basename = os.path.basename(args.gen)
    basename = os.path.splitext(basename)[0]

    gen_selected_path_new = os.path.join(args.save_dir, basename.replace('b0.0_all', f'b{b}_single') + '.txt')

    # select a single summary sentence for each source sentence with length penalty
    os.makedirs(args.save_dir, exist_ok=True)
    g = open(gen_selected_path_new, 'w')

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
            ssa = sorted(ssa, key=lambda x: float(x[1].split()[0]) / (len(x[0].split()) - lp) ** b, reverse=True)
            # float(x[1].split()[0]) for the combined score
            # float(x[1].split()[1]) for the contextual matching score
            # float(x[1].split()[2]) for the language model score

            # if arts[count - 1] == '<unk>':
            #     g.write('\n')
            # else:
            #     if len(ssa[0][0].split()) <= 1:
            #         g.write(arts[count - 1])
            #         g.write('\n')
            #         cp_rate.append(1)
            #     else:
            #         g.write(' '.join(ssa[0][0].split()[:-1]))
            #         g.write('\n')
            #         cp_rate.append(copy_rate(' '.join(ssa[0][0].split()[:-1]), arts[count - 1]))

            if len(ssa[0][0].split()) <= 1:
                # blank line: directly copy the source for summary
                g.write(arts[count - 1])
                g.write('\n')
                cp_rate.append(1)
                comp_rate.append(1)
                lens.append(len(arts[count - 1].split()))
            else:
                g.write(' '.join(ssa[0][0].split()[:-1]))  # do not include the last token, which is to match the <eos>
                g.write('\n')
                cp_rate.append(copy_rate(' '.join(ssa[0][0].split()[:-1]), arts[count - 1]))
                comp_rate.append(len(ssa[0][0].split()[:-1]) / len(arts[count - 1].split()))
                lens.append(len(ssa[0][0].split()[:-1]))
            i = j
        j += 1

    g.close()

    # print out the results and calculate the Rouge scores
    os.system('sed -i "s/<unk>/UNK/g" ' + gen_selected_path_new)
    print('copy rate: %f' % (sum(cp_rate) / len(cp_rate)))
    print('compression rate: %f' % (sum(comp_rate) / len(comp_rate)))
    print('average summary length: %f' % (sum(lens) / len(lens)))
    os.system('files2rouge ' + gen_selected_path_new + ' ' + args.ref)
