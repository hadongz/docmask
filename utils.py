
def change_last_line():
    with open(f"./labels/labels_with_corners.txt", "r") as f:
        with open(f"./labels/new_train_labels.txt", "w") as fn:
            for line in f:
                split = line.split(',')[:2]
                if 'non' in split[0]:
                    split += ['0\n']
                else:
                    split += ['1\n']
                fn.write(','.join(split))

def cat_model_summary(s):
    with open('./logs/model_summary.txt','w') as f:
        print(s, file=f)