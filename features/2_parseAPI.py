import os
from tqdm import tqdm

def parse_label(line): #parse label to api
    str = ""
    nclass = ""
    nfunction = ""
    if(line.startswith("L")):
        str = line[1:]
    else:
        str = line
    #print str
    str = str.rpartition("->")
    nclass = str[0]
    nfunction = str[2] #setLayoutStateDirection(I)V
    #print nfunction
    nclass = nclass[0:-1]
    nclass = nclass.replace('/','.')
    nfunction = nfunction.partition("(")[0]
    # nfunction = nfunction.rpartition(")")
    # type = nfunction[2].rpartition('[')[0]
    # nfunction = type + nfunction[0] + nfunction[1]
    api = nclass + "." +nfunction
    return api.replace("<","").replace(">","")

if __name__ == '__main__':
    # 📁 输入文件夹（可以包含子文件夹）
    input_dir = r"/home/qly/mycode/malradar/download_API"

    # 📂 输出文件夹
    output_dir = r"/home/qly/mycode/malradar/processed_API"
    os.makedirs(output_dir, exist_ok=True)

    # 🔍 收集所有 txt 文件路径
    txt_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    print(f"📄 共发现 {len(txt_files)} 个 txt 文件，开始处理...\n")

    # ✅ tqdm 进度条
    for input_file_path in tqdm(txt_files, desc="Processing TXT files", ncols=100):
        # 相对路径保持文件夹层级
        rel_path = os.path.relpath(input_file_path, input_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 修改输出文件名（加后缀）
        base, ext = os.path.splitext(output_file_path)
        output_file_path = base + "_processed" + ext

        # 文件读取 + 写入处理
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
                open(output_file_path, 'w', encoding='utf-8') as output_file:

            for line in input_file:
                output_file.write(parse_label(line.strip()) + "\n")

    print("\n🎉 所有文件处理完成！输出路径：", output_dir)

