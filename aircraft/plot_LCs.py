import os
import matplotlib.pyplot as plt
fns = [r'C:\Users\Admin\few_shot_learning\aircraft\plots\\' + fn for fn in os.listdir(r'C:\Users\Admin\few_shot_learning\aircraft\plots') if fn.find('.txt') != -1]

fig = plt.figure()
for fn in fns:
    with open(fn, 'r') as f:
        contents = f.read()
        try:
            metrics  = eval(contents)
            val_loss = metrics['val_loss']
            loss = metrics['loss']
            X = range(1, len(val_loss) + 1)
            plt.plot(X, val_loss, label='val_loss ' + fn[52:-4])
            # plt.plot(X, loss, label='loss ' +fn[52:-4])

        except:
            pass
plt.legend()

plt.savefig(r'C:\Users\Admin\few_shot_learning\aircraft\plots\all_LCs.pdf')

plt.show()