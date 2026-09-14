"""
PlantGenome Agent - 数据库初始化脚本

用法：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/init_db.py

功能：
    1. 创建所有表（users, documents, document_chunks, chat_sessions, messages, tasks）
    2. 可选：创建默认测试用户
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import init_db, get_session, engine, get_db_url
from app.models import User
from app.core.security import get_password_hash


def create_default_user():
    """创建默认测试用户。"""
    with get_session() as db:
        existing = db.query(User).filter(User.username == "admin").first()
        if existing:
            print("  默认用户已存在，跳过")
            return

        user = User(
            username="admin",
            email="admin@plantgenome.local",
            hashed_password=get_password_hash("admin123"),
            is_active=True,
        )
        db.add(user)
        print("  ✅ 创建默认用户: admin / admin123")


def main():
    print("=" * 60)
    print("PlantGenome Agent - 数据库初始化")
    print("=" * 60)

    print(f"\n数据库连接: {get_db_url()}")

    # 1. 创建表
    print("\n[1/2] 创建数据表...")
    init_db()

    # 2. 创建默认用户
    print("\n[2/2] 创建默认测试用户...")
    create_default_user()

    print("\n" + "=" * 60)
    print("✅ 数据库初始化完成！")
    print("=" * 60)
    print("\n默认账号:")
    print("  用户名: admin")
    print("  密码:   admin123")
    print("\n注意：生产环境请修改默认密码！")


if __name__ == "__main__":
    main()
