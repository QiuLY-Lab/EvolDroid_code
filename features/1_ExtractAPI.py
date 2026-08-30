import os
from tqdm import tqdm
from androguard.misc import AnalyzeAPK

# 📁 APK 文件夹路径
apk_root_dir = r"/home/qly/mycode/CADE_new"

# 📂 输出保存的文件夹路径
output_dir = r"/home/qly/mycode/CADE_new_API"
os.makedirs(output_dir, exist_ok=True)

# 🔍 递归获取所有 .apk 文件路径
apk_files = []
for root, _, files in os.walk(apk_root_dir):
    for f in files:
        if f.endswith(".apk"):
            apk_files.append(os.path.join(root, f))

print(f"🔍 共检测到 {len(apk_files)} 个 APK 文件，开始提取 API 调用...\n")

# tqdm 进度条显示
for apk_path in tqdm(apk_files, desc="正在提取 APK API 调用", unit="apk"):
    apk_name = os.path.basename(apk_path)

    # 输出文件名（和 APK 同名）
    output_file = os.path.join(output_dir, os.path.splitext(apk_name)[0] + "_api.txt")

    # 🚫 如果文件已存在，则跳过
    if os.path.exists(output_file):
        tqdm.write(f"⚡ 已存在，跳过：{apk_name}")
        continue

    try:
        # 分析 APK
        a, d, dx = AnalyzeAPK(apk_path)
        apis = set()

        # 遍历方法获取调用关系
        for meth in dx.get_methods():
            for _, call, _ in meth.get_xref_to():
                apis.add(f"{call.class_name}->{call.name}")

        # 写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            for api in sorted(apis):
                f.write(api + "\n")

    except Exception as e:
        print(f"\n❌ 处理 {apk_name} 时出错：{e}")

print("\n✅ 所有 APK 提取完成！结果已保存到：", output_dir)
