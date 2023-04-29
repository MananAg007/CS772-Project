f = open('./snli_1.0/snli_1.0_train.jsonl')
format = ",".join(f.readlines())

out = open("./snli_1.0/snli_1.0_train_format.jsonl","w")
out.write("["+format+"]")

f = open('./snli_1.0/snli_1.0_dev.jsonl')
format = ",".join(f.readlines())

out = open("./snli_1.0/snli_1.0_dev_format.jsonl","w")
out.write("["+format+"]")

f = open('./snli_1.0/snli_1.0_test.jsonl')
format = ",".join(f.readlines())

out = open("./snli_1.0/snli_1.0_test_format.jsonl","w")
out.write("["+format+"]")