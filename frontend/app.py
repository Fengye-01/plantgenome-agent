"""
PlantGenome Agent - Streamlit 前端 v2.0

新增功能：
- 用户注册/登录（JWT 认证）
- 历史会话列表
- 消息持久化
- PDF 上传异步处理 + 任务状态轮询
- Agent 执行过程可视化

运行：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    streamlit run frontend/app.py --server.port 8502 --server.fileWatcherType none
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
import pandas as pd

# Docker Compose 会传入容器内的后端地址；本地启动时使用 8001。
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PlantGenome Agent",
    page_icon="🌱",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════
# 会话状态初始化
# ═══════════════════════════════════════════════════════════

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sessions" not in st.session_state:
    st.session_state.sessions = []

if "document_delete_confirm_id" not in st.session_state:
    st.session_state.document_delete_confirm_id = None


# ═══════════════════════════════════════════════════════════
# API 辅助函数
# ═══════════════════════════════════════════════════════════

def get_headers():
    """获取带 JWT 的请求头。"""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def api_login(username: str, password: str):
    """登录。"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            data={"username": username, "password": password},
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.current_user = data["user"]
            return True, None
        return False, resp.json().get("detail", "登录失败")
    except Exception as e:
        return False, f"连接服务器失败: {str(e)}"


def api_register(username: str, email: str, password: str):
    """注册。"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        if resp.status_code == 201:
            return True, None
        return False, resp.json().get("detail", "注册失败")
    except Exception as e:
        return False, f"连接服务器失败: {str(e)}"


def api_chat(message: str, session_id: int = None):
    """发送聊天消息。"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"message": message, "session_id": session_id},
            headers=get_headers(),
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "请求失败")
    except Exception as e:
        return None, f"连接服务器失败: {str(e)}"


def api_get_sessions():
    """获取会话列表。"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/chat/sessions",
            headers=get_headers(),
        )
        if resp.status_code == 200:
            return resp.json(), None
        return [], resp.json().get("detail", "获取失败")
    except Exception as e:
        return [], f"连接服务器失败: {str(e)}"


def api_get_messages(session_id: int):
    """获取会话历史消息。"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/chat/sessions/{session_id}/messages",
            headers=get_headers(),
        )
        if resp.status_code == 200:
            return resp.json(), None
        return [], resp.json().get("detail", "获取失败")
    except Exception as e:
        return [], f"连接服务器失败: {str(e)}"


def api_upload_pdf(file):
    """上传 PDF（异步）。"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        resp = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files,
            headers=get_headers(),
        )
        if resp.status_code in (200, 202):
            return resp.json(), None
        return None, resp.json().get("detail", "上传失败")
    except Exception as e:
        return None, f"连接服务器失败: {str(e)}"


def api_get_task(task_id: int):
    """查询任务状态。"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/tasks/{task_id}",
            headers=get_headers(),
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "查询失败")
    except Exception as e:
        return None, f"连接服务器失败: {str(e)}"


def api_get_documents():
    """获取当前用户的知识库文档。"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/documents",
            headers=get_headers(),
        )
        if resp.status_code == 200:
            return resp.json(), None
        return [], resp.json().get("detail", "获取文档失败")
    except Exception as e:
        return [], f"连接服务器失败: {str(e)}"


def api_delete_document(document_id: int):
    """删除当前用户的一篇知识库文档。"""
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/api/documents/{document_id}",
            headers=get_headers(),
        )
        if resp.status_code == 204:
            return True, None
        return False, resp.json().get("detail", "删除文档失败")
    except Exception as e:
        return False, f"连接服务器失败: {str(e)}"


def api_pubmed_search(keyword: str, max_results: int = 5):
    """从 PubMed 检索文献并自动下载入库（异步）。"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/documents/pubmed-search",
            json={"keyword": keyword, "max_results": max_results},
            headers=get_headers(),
        )
        if resp.status_code in (200, 202):
            return resp.json(), None
        return None, resp.json().get("detail", "检索失败")
    except Exception as e:
        return None, f"连接服务器失败: {str(e)}"


# ═══════════════════════════════════════════════════════════
# 登录/注册页面
# ═══════════════════════════════════════════════════════════

def show_auth_page():
    """显示登录/注册页面。"""
    st.title("🌱 PlantGenome Agent")
    st.caption("面向植物比较基因组研究的 PDF-RAG + Tool Calling 智能体")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["登录", "注册"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                submit = st.form_submit_button("登录", use_container_width=True)

                if submit:
                    if not username or not password:
                        st.error("请输入用户名和密码")
                    else:
                        with st.spinner("登录中..."):
                            success, error = api_login(username, password)
                            if success:
                                st.success("登录成功！")
                                st.rerun()
                            else:
                                st.error(f"登录失败: {error}")

        with tab2:
            with st.form("register_form"):
                reg_username = st.text_input("用户名", key="reg_username")
                reg_email = st.text_input("邮箱", key="reg_email")
                reg_password = st.text_input("密码", type="password", key="reg_password")
                reg_confirm = st.text_input("确认密码", type="password", key="reg_confirm")
                submit = st.form_submit_button("注册", use_container_width=True)

                if submit:
                    if not all([reg_username, reg_email, reg_password]):
                        st.error("请填写所有字段")
                    elif reg_password != reg_confirm:
                        st.error("两次密码不一致")
                    elif len(reg_password) < 6:
                        st.error("密码至少 6 位")
                    else:
                        with st.spinner("注册中..."):
                            success, error = api_register(reg_username, reg_email, reg_password)
                            if success:
                                st.success("注册成功！请切换到登录页登录")
                            else:
                                st.error(f"注册失败: {error}")

    st.divider()
    st.caption("默认测试账号: admin / admin123（需先运行 python scripts/init_db.py 初始化数据库）")


# ═══════════════════════════════════════════════════════════
# 展示函数
# ═══════════════════════════════════════════════════════════

def display_execution_log(execution_log: list):
    """展示 Agent 执行过程。"""
    if not execution_log:
        return
    with st.expander("🔍 Agent 执行过程", expanded=False):
        total_latency = 0
        for i, log in enumerate(execution_log, 1):
            node = log.get("node", "?")
            status = log.get("status", "?")
            latency = log.get("latency", 0)
            if isinstance(latency, (int, float)):
                total_latency += latency
            icon = {"router": "🧭", "tool": "🔧", "answer": "💬"}.get(node, "📍")
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1:
                    st.markdown(f"**{icon} {i}. {node.capitalize()}**")
                with c2:
                    if node == "router":
                        st.markdown(f"**意图**: `{log.get('intent', '?')}` | **工具**: `{log.get('tool_name', '无')}`")
                    elif node == "tool":
                        st.markdown(f"**工具**: `{log.get('tool', '?')}`")
                    elif node == "answer":
                        st.markdown(f"**回答长度**: {log.get('answer_length', '?')} 字")
                with c3:
                    st.markdown(f"**{latency}s**" if isinstance(latency, (int, float)) else f"**{latency}**")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("总节点数", len(execution_log))
        with c2:
            st.metric("总耗时", f"{round(total_latency, 1)}s")


def display_tool_result(tool_name: str, tool_result, intent: str):
    """展示工具结果。"""
    if tool_result is None:
        return
    if isinstance(tool_result, dict) and "error" in tool_result:
        st.error(f"⚠️ 工具执行失败: {tool_result['error']}")
        return
    with st.expander("📊 工具执行结果", expanded=True):
        if intent == "fasta_analysis" and isinstance(tool_result, dict):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("序列数量", tool_result.get("num_sequences", 0))
            c2.metric("总长度", tool_result.get("total_length", 0))
            c3.metric("平均长度", tool_result.get("avg_length", 0))
            c4.metric("GC 含量", f"{tool_result.get('gc_content', 0)*100:.1f}%")
            dist = tool_result.get("length_distribution", {})
            if dist:
                st.markdown("**长度分布**")
                st.bar_chart(pd.DataFrame({"区间": list(dist.keys()), "数量": list(dist.values())}).set_index("区间"))
        elif intent == "cpg_scan" and isinstance(tool_result, list):
            islands = [x for x in tool_result if isinstance(x, dict) and "start" in x]
            if islands:
                st.metric("CpG 岛数量", len(islands))
                st.dataframe(pd.DataFrame(islands), use_container_width=True)
            else:
                for item in tool_result:
                    if isinstance(item, dict):
                        if "warning" in item:
                            st.warning(item["warning"])
                        elif "result" in item:
                            st.info(item["result"])
        elif intent == "pipeline_suggest" and isinstance(tool_result, dict):
            st.markdown(f"**{tool_result.get('pipeline_name', '推荐流程')}**")
            for step in tool_result.get("steps", []):
                if isinstance(step, dict):
                    with st.container(border=True):
                        st.markdown(f"### 步骤 {step.get('step', '?')}: {step.get('name', '?')}")
                        st.markdown(f"🔧 工具: `{step.get('tool', '?')}`")
                        st.markdown(f"📥 输入: {step.get('input', '?')}")
                        st.markdown(f"📤 输出: {step.get('output', '?')}")


def display_sources(sources: list):
    """展示来源引用。"""
    if not sources:
        return
    with st.expander("📚 来源引用", expanded=False):
        for i, s in enumerate(sources, 1):
            if isinstance(s, dict):
                st.markdown(f"**[{i}] {s.get('filename', '?')}** (p.{s.get('page_num', '?')})")
                if s.get("snippet"):
                    st.markdown(f"> {s['snippet'][:200]}")
                st.divider()


def format_file_size(size: int) -> str:
    """将字节数格式化为适合目录展示的大小。"""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def display_document_library():
    """在侧边栏展示当前用户的知识库文档。"""
    title_col, refresh_col = st.columns([4, 1])
    with title_col:
        st.header("📚 我的知识库")
    with refresh_col:
        if st.button("↻", key="refresh_documents", help="刷新文档列表"):
            st.rerun()

    documents, error = api_get_documents()
    if error:
        st.error(error)
        return
    if not documents:
        st.caption("尚未上传文档")
        return

    status_labels = {
        "pending": ("⏳", "等待处理"),
        "running": ("🔄", "处理中"),
        "completed": ("✅", "已完成"),
        "failed": ("❌", "处理失败"),
    }

    for document in documents:
        document_id = document["id"]
        status_icon, status_text = status_labels.get(
            document.get("status"),
            ("•", document.get("status", "未知")),
        )
        filename = document.get("filename", "未命名文档")
        with st.expander(f"{status_icon} {filename}", expanded=False):
            st.caption(
                f"{status_text} · {format_file_size(document.get('file_size', 0))} · "
                f"{document.get('chunk_count', 0)} chunks"
            )
            created_at = document.get("created_at")
            if created_at:
                st.caption(f"上传时间：{created_at[:16].replace('T', ' ')}")
            if document.get("error_message"):
                st.error(document["error_message"])

            if st.session_state.document_delete_confirm_id == document_id:
                st.warning("删除后将同时移除原文件、分块记录和向量，无法恢复。")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button(
                        "确认删除",
                        key=f"confirm_delete_document_{document_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        success, delete_error = api_delete_document(document_id)
                        if success:
                            st.session_state.document_delete_confirm_id = None
                            st.toast("文档已删除")
                            st.rerun()
                        st.error(delete_error)
                with cancel_col:
                    if st.button(
                        "取消",
                        key=f"cancel_delete_document_{document_id}",
                        use_container_width=True,
                    ):
                        st.session_state.document_delete_confirm_id = None
                        st.rerun()
            elif st.button(
                "删除文档",
                key=f"delete_document_{document_id}",
                use_container_width=True,
            ):
                st.session_state.document_delete_confirm_id = document_id
                st.rerun()


# ═══════════════════════════════════════════════════════════
# 主应用
# ═══════════════════════════════════════════════════════════

def show_main_app():
    """显示主应用（已登录）。"""
    user = st.session_state.current_user

    # 顶部栏
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.title("🌱 PlantGenome Agent")
    with col2:
        st.markdown(f"<div style='text-align:right; padding-top:20px;'>👤 {user.get('username', '')}</div>", unsafe_allow_html=True)
    with col3:
        if st.button("退出登录", key="logout_btn"):
            st.session_state.access_token = None
            st.session_state.current_user = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

    # 侧边栏
    with st.sidebar:
        st.header("📁 会话")
        if st.button("➕ 新会话", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        # 会话列表
        sessions, _ = api_get_sessions()
        st.session_state.sessions = sessions
        for s in sessions:
            label = f"💬 {s.get('title', '未命名')[:30]}"
            if st.button(label, key=f"session_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                messages, _ = api_get_messages(s["id"])
                st.session_state.messages = messages
                st.rerun()

        st.divider()
        st.header("📄 文档上传")
        uploaded_file = st.file_uploader("上传 PDF", type=["pdf"], key="pdf_uploader")
        if uploaded_file:
            if st.button("上传并解析", use_container_width=True):
                with st.spinner("上传中..."):
                    result, error = api_upload_pdf(uploaded_file)
                    if result:
                        task_id = result.get("task_id")
                        st.success(f"上传成功！任务 ID: {task_id}")
                        # 轮询任务状态（用 placeholder 覆盖更新，避免重复刷屏）
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        task = None
                        for _ in range(60):
                            time.sleep(2)
                            task, _ = api_get_task(task_id)
                            if task:
                                progress = task.get("progress", 0)
                                status = task.get("status", "")
                                status_placeholder.info(f"状态: {status} | 进度: {progress}%")
                                progress_bar.progress(progress / 100)
                                if status in ("completed", "failed"):
                                    break
                        # 清除进度显示，展示最终结果
                        status_placeholder.empty()
                        progress_bar.empty()
                        if task and task.get("status") == "completed":
                            task_result = task.get("result") or {}
                            # result 可能是 JSON 字符串，做兼容处理
                            if isinstance(task_result, str):
                                import json
                                try:
                                    task_result = json.loads(task_result)
                                except Exception:
                                    task_result = {}
                            chunk_count = task_result.get("chunk_count", 0)
                            pages = task_result.get("pages", "?")
                            st.success(f"✅ 处理完成！{pages} 页，{chunk_count} 个 chunks")
                        elif task and task.get("status") == "failed":
                            st.error(f"❌ 处理失败: {task.get('error_message', '未知错误')}")
                    else:
                        st.error(f"上传失败: {error}")

        st.divider()
        display_document_library()

        st.divider()
        st.header("🔍 PubMed 文献检索")
        st.caption("输入关键字，自动从 PubMed 下载文献并入库 RAG")

        pubmed_keyword = st.text_input("搜索关键字", key="pubmed_keyword",
                                        placeholder="如: plant CpG island methylation")
        pubmed_max = st.slider("文献数量", min_value=1, max_value=10, value=3, key="pubmed_max")

        if st.button("🔍 搜索并下载", use_container_width=True, key="pubmed_search_btn"):
            if not pubmed_keyword.strip():
                st.error("请输入搜索关键字")
            else:
                with st.spinner("提交检索任务..."):
                    result, error = api_pubmed_search(pubmed_keyword.strip(), pubmed_max)
                    if result:
                        task_id = result.get("task_id")
                        st.success(f"任务已提交！ID: {task_id}")
                        # 轮询任务状态
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        task = None
                        for _ in range(120):  # 最多等 4 分钟（120 * 2s）
                            time.sleep(2)
                            task, _ = api_get_task(task_id)
                            if task:
                                progress = task.get("progress", 0)
                                status = task.get("status", "")
                                status_placeholder.info(f"状态: {status} | 进度: {progress}%")
                                progress_bar.progress(min(progress / 100, 1.0))
                                if status in ("completed", "failed"):
                                    break
                        status_placeholder.empty()
                        progress_bar.empty()

                        if task and task.get("status") == "completed":
                            task_result = task.get("result") or {}
                            if isinstance(task_result, str):
                                import json
                                try:
                                    task_result = json.loads(task_result)
                                except Exception:
                                    task_result = {}
                            total = task_result.get("total", 0)
                            success = task_result.get("success", 0)
                            failed = task_result.get("failed", 0)
                            st.success(f"✅ 完成！成功 {success}/{total} 篇，失败 {failed} 篇")

                            # 展示每篇文献的结果
                            articles = task_result.get("articles", [])
                            if articles:
                                with st.expander("📚 文献详情", expanded=True):
                                    for art in articles:
                                        status_icon = "✅" if art.get("status") == "completed" else "❌"
                                        source = art.get("source", "")
                                        source_label = "全文" if source == "full_text" else ("摘要" if source == "abstract" else "")
                                        st.markdown(
                                            f"{status_icon} **PMID: {art.get('pmid', '?')}** "
                                            f"{'(`' + source_label + '`)' if source_label else ''}"
                                        )
                                        st.markdown(f"  {art.get('title', '?')[:80]}")
                                        if art.get("status") == "completed":
                                            st.markdown(f"  📊 {art.get('chunk_count', 0)} chunks")
                                        elif art.get("error"):
                                            st.markdown(f"  ⚠️ {art.get('error')[:80]}")
                                        st.divider()
                            st.info("💡 文献已入库，现在可以在聊天中提问相关内容了")
                        elif task and task.get("status") == "failed":
                            st.error(f"❌ 检索失败: {task.get('error_message', '未知错误')}")
                        else:
                            st.warning("⏳ 任务仍在进行中，可稍后在任务列表查看")
                    else:
                        st.error(f"检索失败: {error}")

    # 主区域：聊天
    if not st.session_state.current_session_id:
        st.info("👈 从左侧选择一个会话，或直接开始新对话")

    # 显示消息
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))
            if msg.get("role") == "assistant":
                if msg.get("tool_name") and msg.get("tool_result"):
                    display_tool_result(msg["tool_name"], msg["tool_result"], msg.get("tool_name", ""))
                if msg.get("sources"):
                    display_sources(msg["sources"])
                if msg.get("execution_log"):
                    display_execution_log(msg["execution_log"])

    # 聊天输入
    if prompt := st.chat_input("输入你的问题..."):
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 回答
        with st.chat_message("assistant"):
            with st.spinner("🧭 Agent 正在思考..."):
                result, error = api_chat(prompt, st.session_state.current_session_id)
                if error:
                    st.error(f"错误: {error}")
                else:
                    answer = result.get("answer", "")
                    session_id = result.get("session_id")
                    sources = result.get("sources", [])
                    tool_name = result.get("tool_name")
                    tool_result = result.get("tool_result")
                    execution_log = result.get("execution_log")

                    st.session_state.current_session_id = session_id
                    st.markdown(answer)

                    if tool_name and tool_result:
                        display_tool_result(tool_name, tool_result, tool_name)
                    if sources:
                        display_sources(sources)
                    if execution_log:
                        display_execution_log(execution_log)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "tool_name": tool_name,
                        "tool_result": tool_result,
                        "execution_log": execution_log,
                    })


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

if st.session_state.access_token is None:
    show_auth_page()
else:
    show_main_app()
