f = open('./snli_1.0/snli_1.0_train.jsonl')
format = ",".join(f.readlines())

out = open("./snli_1.0/snli_1.0_train_format.jsonl","w")
out.write("["+format+"]")
