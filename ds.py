# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import glob
import json
HISTORY_DIR = "history"
role_content="AI 分析提示词（Prompt）\
你是一位专业的人因工程与上肢生物力学分析专家，请基于以下提供的手腕运动轨迹量化数据，结合医学背景，对用户是否存在“鼠标手”（腕管综合征）风险进行科学评估，并给出可操作的改进建议。\
背景知识（供你参考）\
“鼠标手”在临床上称为腕管综合征（Carpal Tunnel Syndrome, CTS），是最常见的周围神经卡压性疾患。\
它主要由于长时间重复性手部动作（如使用鼠标、键盘）导致手腕反复屈伸、尺偏/桡偏或固定姿势，使正中神经在腕管内受压，引发疼痛、麻木、酸胀、夜间麻醒、握力下降等症状。\
高风险动作特征包括:\
高频小幅重复移动（低幅度但高频率）\
手腕极端角度保持（如过度背伸或尺偏）\
缺乏休息与姿势变换\
动作方向突变频繁（急停、急转）\
输入数据说明\
我提供一个 JSON 文件，包含以下三个时间序列指标（按视频帧顺序记录，每帧一个值）：\
euclid：相邻帧之间手腕像素位移（反映瞬时移动速度/幅度）\
cos：连续两段位移向量的余弦相似度（范围 [-1, 1]；1=同向平滑，-1=反向突变，0=直角转向）\
angle：连续两段位移的转向夹角（单位：度，0°~180°；值越大表示方向突变越剧烈）\
数据来源于一段真实工作场景视频（如使用鼠标操作），通过 YOLO 姿态估计模型提取手腕关键点（keypoint 0）轨迹后计算得出。\
请你完成以下分析任务：\
运动模式识别\
用户的手腕运动以高频小幅移动为主，还是低频大幅移动？\
是否存在长时间几乎不动（euclid ≈ 0 持续多帧）？\
动作方向是否高度重复（cos 稳定接近 1）？还是频繁急转（angle 频繁 > 60°）？\
风险等级评估\
综合上述特征，判断用户当前操作模式的腕管综合征风险等级（低 / 中 / 高），并说明依据。\
医学-工程结合建议\
针对识别出的问题，给出具体、可执行的人机工学改善建议，例如：\
鼠标使用姿势调整\
工作-休息节奏（如每 X 分钟活动手腕 Y 秒）\
推荐的辅助工具（垂直鼠标、腕托等）\
简单的手腕放松拉伸动作（附文字说明）\
请用清晰、简洁、非技术性语言输出，面向普通办公人员，避免医学术语堆砌，重点突出** actionable insights **（可行动的建议）。"
client = OpenAI(
    api_key="xxxxxxxxxxxxxxxxxxx", # TODO  replace with your DeepSeek API key
    base_url="https://api.deepseek.com")


def ds():
    
    json_files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{HISTORY_DIR}' directory.")


    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading latest metrics file: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    content = json.dumps(data, indent=2, ensure_ascii=False)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": role_content},
            {"role": "user", 
            "content": f"以下是用户的手腕运动轨迹数据（JSON格式）：\n\n{content}"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)