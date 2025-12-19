# hand_disease_recognition
Quick start
1.克隆本仓库\
` git clone https://github.com/SDX123321/hand_disease_recognition.git`\
2. 安装对应依赖\
`pip install -r requirements.txt`\
3. 拍摄操作电脑使鼠标手的手腕运动视频（30s以上为宜，光照充足）\
4. 编辑config.py\
将`VIDEO_PATH`值更改为拍摄视频的路径名\
5. 填写DEEPSEEK_API\
将`ds.py`中的`api_key`写为自己的api密钥,请在`https://platform.deepseek.com/usage`处申请\
6. 运行本项目\
执行`python main.py`以完整运行本项目\
