import os
import random


random.seed(2)
rate_of_train = 0.8

bird_names = [item for item in os.listdir(".") if os.path.splitext(item)[-1]==""]

with open("classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(bird_names))+"\n")

f.close()

all_f = open("all.txt", "w", encoding="utf-8")
train_f = open("train.txt", "w", encoding="utf-8")
val_f = open("val.txt", "w", encoding="utf-8")

for bird_name in bird_names:
    if os.path.isdir(bird_name):
        filenames = os.listdir(bird_name)
        for filename in filenames:
            all_f.write("images/{}/{},{}\n".format(bird_name, filename, bird_name))
            if random.random() < rate_of_train:
                train_f.write("images/{}/{},{}\n".format(bird_name, filename, bird_name))
            else:
                val_f.write("images/{}/{},{}\n".format(bird_name, filename, bird_name))
print("Done!")
all_f.close()
train_f.close()
val_f.close()
