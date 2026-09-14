"""
测试 PDF 上传 + 异步处理 + 聊天问答
"""
import time
import requests

API_BASE = "http://localhost:8000"

# 1. 登录
print("=" * 60)
print("步骤 1: 登录")
print("=" * 60)
r = requests.post(f"{API_BASE}/api/auth/login", data={
    "username": "admin",
    "password": "admin123",
})
r.raise_for_status()
token = r.json()["access_token"]
print(f"✅ 登录成功，token: {token[:30]}...")

headers = {"Authorization": f"Bearer {token}"}

# 2. 上传 PDF
print("\n" + "=" * 60)
print("步骤 2: 上传 PDF")
print("=" * 60)
pdf_path = "data/pdfs/Yang - 2007 - PAML 4 Phylogenetic analysis by maximum likelihood.pdf"
with open(pdf_path, "rb") as f:
    r = requests.post(
        f"{API_BASE}/api/documents/upload",
        files={"file": ("PAML_test_upload.pdf", f, "application/pdf")},
        headers=headers,
    )
r.raise_for_status()
upload_result = r.json()
print(f"✅ 上传成功")
print(f"  任务 ID: {upload_result['task_id']}")
print(f"  文档 ID: {upload_result['document_id']}")
print(f"  状态: {upload_result['status']}")

# 3. 轮询任务状态
print("\n" + "=" * 60)
print("步骤 3: 轮询任务状态")
print("=" * 60)
task_id = upload_result["task_id"]
doc_id = upload_result["document_id"]

for i in range(40):
    time.sleep(3)
    r = requests.get(f"{API_BASE}/api/tasks/{task_id}", headers=headers)
    r.raise_for_status()
    task = r.json()
    print(f"  第 {i+1} 次: status={task['status']}, progress={task['progress']}%")

    if task["status"] == "completed":
        print(f"\n✅ 任务完成！")
        print(f"  结果: {task.get('result', {})}")
        break
    if task["status"] == "failed":
        print(f"\n❌ 任务失败！")
        print(f"  错误: {task.get('error_message', '未知错误')}")
        break

# 4. 验证文档状态
print("\n" + "=" * 60)
print("步骤 4: 验证文档状态")
print("=" * 60)
r = requests.get(f"{API_BASE}/api/documents/{doc_id}", headers=headers)
r.raise_for_status()
doc = r.json()
print(f"  文档名: {doc['filename']}")
print(f"  状态: {doc['status']}")
print(f"  Chunk 数量: {doc['chunk_count']}")
print(f"  文件大小: {doc['file_size'] / 1024:.1f} KB")

# 5. 测试聊天（基于刚上传的文档）
print("\n" + "=" * 60)
print("步骤 5: 测试聊天问答")
print("=" * 60)
r = requests.post(
    f"{API_BASE}/api/chat",
    json={"message": "PAML 中的 omega 值是什么意思？"},
    headers=headers,
    timeout=120,
)
r.raise_for_status()
chat_result = r.json()
print(f"✅ 聊天成功")
print(f"  会话 ID: {chat_result['session_id']}")
print(f"  工具: {chat_result.get('tool_name', '无')}")
print(f"  回答长度: {len(chat_result['answer'])} 字")
print(f"  来源数量: {len(chat_result.get('sources', []))}")
print(f"\n  回答前 200 字:")
print(f"  {chat_result['answer'][:200]}")
if chat_result.get("sources"):
    print(f"\n  来源:")
    for s in chat_result["sources"]:
        print(f"    - {s.get('filename', '?')} p.{s.get('page_num', '?')}")

print("\n" + "=" * 60)
print("✅ 全部测试通过！")
print("=" * 60)
