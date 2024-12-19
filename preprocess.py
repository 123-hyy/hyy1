import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = 'D:/大三上/模式识别/pythonProject/IMCS-DAC'
train_set = load_json(os.path.join(data_dir, 'IMCS-DAC_train.json'))
dev_set = load_json(os.path.join(data_dir, 'IMCS-DAC_dev.json'))
test_set = load_json(os.path.join(data_dir, 'IMCS-DAC_test.json'))

saved_path = os.path.join(data_dir, 'processed_data')
os.makedirs(saved_path, exist_ok=True)

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}

id2tag = {}
for tag in tag2id:
    id2tag[tag2id[tag]] = tag


def make_tag(path):
    with open(path, 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write(tag + '\n')

def parse_data(sent: dict, check_dialogue_act: bool):
    try:
        x = sent['speaker'] + '：' + sent['sentence']
        if check_dialogue_act:
            y = tag2id.get(sent['dialogue_act'])
            assert sent['dialogue_act'] in tag2id
        else:
            y=15
        # print(f"x = {x}")
        # print(f"y = {str(y)}")

        out = x + '\t' + str(y) + '\n'
        return out
    except Exception as e:
        print(e)
    return None

def make_data(samples, path, check_dialogue_act: bool):
    out = ''
    all_size = 0
    data_size = 0
    for pid, sample in samples.items():
        # print(f"pid = {pid}")
        # print(sample)
        for sent in sample:
            # print(sent)
            if type(sent)==list and len(sent)==2:
                sample = sent[1]
                for ss in sample:
                    try:
                        data = parse_data(ss, check_dialogue_act)
                        if data:
                            out += data
                            data_size += 1
                    except Exception as e:
                        print(e)
                break
            all_size += 1
            # print(f"data size = {data_size}")
            try:
                data = parse_data(sent, check_dialogue_act)
                if data:
                    out += data
                    data_size += 1
            except Exception as e:
                print(e)
    print(f"data size = {data_size}")
    print(f"all size = {all_size}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out


make_tag(os.path.join(saved_path, 'class.txt'))

# make_data(train_set, os.path.join(saved_path, 'train.txt'), True)
# make_data(dev_set, os.path.join(saved_path, 'dev.txt'), True)
make_data(test_set, os.path.join(saved_path, 'test.txt'), False)
