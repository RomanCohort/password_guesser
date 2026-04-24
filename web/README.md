# Password Guesser - Web Interface

基于 FastAPI 的 Web 前端界面。

## 启动

```bash
cd password_guesser
uvicorn web.app:app --reload --port 8000
```

然后访问 http://localhost:8000

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/status` | GET | 系统状态 |
| `/api/load_model` | POST | 加载模型 |
| `/api/configure_llm` | POST | 配置 LLM |
| `/api/extract` | POST | 提取特征 |
| `/api/generate` | POST | 生成密码 |
| `/api/score` | POST | 评分单个密码 |
| `/api/methods` | GET | 获取生成方法列表 |

## 使用流程

1. **配置 LLM**: 输入 DeepSeek API Key
2. **输入目标信息**: 填写目标的个人信息
3. **选择生成方法**:
   - Sampling: 标准采样
   - Beam Search: 搜索最优路径
   - Diverse Beam: 多样化输出
   - Typical: 基于信息熵的采样
   - Contrastive: 避免重复
4. **点击生成**: 等待结果
5. **点击密码卡片**: 复制到剪贴板

## 文件结构

```
web/
├── app.py                 # FastAPI 应用
├── templates/
│   └── index.html         # 主页面模板
└── static/
    ├── css/
    │   └── style.css      # 样式表
    └── js/
        └── app.js         # 前端逻辑
```

## API 文档

启动后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
