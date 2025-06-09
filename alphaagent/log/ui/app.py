import argparse
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files as rfiles
from pathlib import Path
from typing import Callable, Type
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state
from streamlit_theme import st_theme
import hashlib
import json

from alphaagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from alphaagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from alphaagent.components.coder.model_coder.evaluators import ModelSingleFeedback
from alphaagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from alphaagent.core.proposal import Hypothesis, HypothesisFeedback
from alphaagent.core.scenario import Scenario
from alphaagent.log.base import Message
from alphaagent.log.storage import FileStorage
from alphaagent.log.ui.qlib_report_figure import report_figure
# from alphaagent.scenarios.data_mining.experiment.model_experiment import DMModelScenario
# from alphaagent.scenarios.general_model.scenario import GeneralModelScenario
# from alphaagent.scenarios.kaggle.experiment.scenario import KGScenario
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario, QlibAlphaAgentScenario
from alphaagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from alphaagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

import requests
from datetime import datetime
import time

# 设置页面配置
st.set_page_config(layout="wide", page_title="AlphaAgent", page_icon="🎓", initial_sidebar_state="expanded")

# 用户文件路径
USERS_FILE = "users.json"

# 默认用户配置
DEFAULT_USERS = {
    "admin": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",  # password: hello
    "user1": "fcf730b6d95236ecd3c9fc2d92d7b6b2bb061514961aec041d6c7a7192f592e4",  # password: secret123
    "demo": "d3ad9315b7be5dd53b31a273b3b3aba5defe700808305aa16a3062b76658a791",   # password: demo123
}

def load_users():
    """从文件加载用户数据"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 如果文件不存在，创建默认用户文件
            save_users(DEFAULT_USERS)
            return DEFAULT_USERS.copy()
    except Exception as e:
        st.error(f"Failed to load user data: {e}")
        return DEFAULT_USERS.copy()

def save_users(users_dict):
    """保存用户数据到文件"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save user data: {e}")
        return False

def hash_password(password: str) -> str:
    """对密码进行SHA256哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(username: str, password: str) -> bool:
    """验证用户名和密码"""
    users = load_users()
    if username in users:
        return users[username] == hash_password(password)
    return False

def is_valid_username(username: str) -> tuple[bool, str]:
    """验证用户名格式"""
    if not username:
        return False, "Username cannot be empty"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 20:
        return False, "Username cannot exceed 20 characters"
    if not username.replace('_', '').replace('-', '').isalnum():
        return False, "Username can only contain letters, numbers, underscores and hyphens"
    return True, ""

def is_valid_password(password: str) -> tuple[bool, str]:
    """验证密码强度"""
    if not password:
        return False, "Password cannot be empty"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if len(password) > 50:
        return False, "Password cannot exceed 50 characters"
    return True, ""

def user_exists(username: str) -> bool:
    """检查用户是否已存在"""
    users = load_users()
    return username in users

def register_user(username: str, password: str) -> tuple[bool, str]:
    """注册新用户"""
    # 验证用户名
    valid_username, username_msg = is_valid_username(username)
    if not valid_username:
        return False, username_msg
    
    # 验证密码
    valid_password, password_msg = is_valid_password(password)
    if not valid_password:
        return False, password_msg
    
    # 检查用户是否已存在
    if user_exists(username):
        return False, "Username already exists, please choose another"
    
    # 添加新用户
    users = load_users()
    users[username] = hash_password(password)
    
    if save_users(users):
        return True, "Registration successful!"
    else:
        return False, "Registration failed, please try again"

def register_page():
    """注册页面"""
    st.markdown("""
    <style>
    /* 注册页面背景 */
    .main .block-container {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    
    /* 注册页面背景装饰圆圈 - 第一层 */
    .main .block-container::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.1) 25%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.05) 35%, transparent 60%),
            radial-gradient(circle at 60% 20%, rgba(17, 153, 142, 0.3) 0%, rgba(17, 153, 142, 0.1) 30%, transparent 50%),
            radial-gradient(circle at 30% 80%, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.08) 40%, transparent 65%),
            radial-gradient(circle at 70% 60%, rgba(56, 239, 125, 0.2) 0%, transparent 45%);
        animation: float 25s ease-in-out infinite;
        z-index: 0;
    }
    
    /* 注册页面背景装饰圆圈 - 第二层 */
    .main .block-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 90% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 35%),
            radial-gradient(circle at 10% 90%, rgba(17, 153, 142, 0.2) 0%, transparent 45%);
        animation: float-reverse 30s ease-in-out infinite;
        z-index: 0;
    }
    
    .welcome-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        position: relative;
        z-index: 2;
    }
    .app-logo {
        font-size: 4rem;
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: pulse 2s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.05); }
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .app-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .register-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .password-rules {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        font-size: 0.9em;
    }
    .password-rules h4 {
        color: #ffffff !important;
        margin-bottom: 1rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .password-rules p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 0.5rem;
    }
    
    /* 注册页面按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(17, 153, 142, 0.4);
        background: linear-gradient(45deg, #38ef7d, #11998e);
    }
    
    /* 注册页面输入框样式 */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: #333;
        font-size: 1rem;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #11998e;
        box-shadow: 0 0 0 3px rgba(17, 153, 142, 0.2);
        background-color: rgba(255, 255, 255, 1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 应用介绍头部
    st.markdown("""
    <div class="welcome-header">
        <div class="app-logo">📝</div>
        <div class="app-title">AlphaAgent</div>
        <div class="app-subtitle">Create Your Account</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 使用streamlit的原生容器
        with st.container(border=True):
            st.markdown('<div class="register-title">📝 User Registration</div>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Please enter username (3-20 characters)")
            password = st.text_input("Password", type="password", placeholder="Please enter password (at least 6 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Please enter password again")
            
            col_register, col_back = st.columns([1, 1])
            with col_register:
                register_clicked = st.form_submit_button("✅ Register", use_container_width=True)
            with col_back:
                back_clicked = st.form_submit_button("🔙 Back to Login", use_container_width=True)
            
            if register_clicked:
                if not username or not password or not confirm_password:
                    st.warning("Please fill in all fields! ⚠️")
                elif password != confirm_password:
                    st.error("Passwords do not match! ❌")
                else:
                    success, message = register_user(username, password)
                    if success:
                        st.success(message + " Please go back to login page to sign in.")
                        # 自动切换回登录页面
                        state.show_register = False
                        st.rerun()
                    else:
                        st.error(message)
            
            if back_clicked:
                state.show_register = False
                st.rerun()
        
        # 密码规则说明
        st.markdown("""
                 <div class="password-rules">
             <h4>📋 Registration Rules</h4>
             <p><strong>Username:</strong> 3-20 characters, only letters, numbers, underscores and hyphens allowed</p>
             <p><strong>Password:</strong> At least 6 characters</p>
         </div>
         """, unsafe_allow_html=True)

def login_page():
    """登录页面"""
    st.markdown("""
    <style>
    /* 登录页面背景 */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    
    /* 背景装饰圆圈 - 第一层 */
    .main .block-container::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.1) 25%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.05) 35%, transparent 60%),
            radial-gradient(circle at 60% 20%, rgba(102, 126, 234, 0.3) 0%, rgba(102, 126, 234, 0.1) 30%, transparent 50%),
            radial-gradient(circle at 30% 80%, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.08) 40%, transparent 65%),
            radial-gradient(circle at 70% 60%, rgba(118, 75, 162, 0.2) 0%, transparent 45%);
        animation: float 25s ease-in-out infinite;
        z-index: 0;
    }
    
    /* 背景装饰圆圈 - 第二层 */
    .main .block-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 90% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 35%),
            radial-gradient(circle at 10% 90%, rgba(102, 126, 234, 0.2) 0%, transparent 45%);
        animation: float-reverse 30s ease-in-out infinite;
        z-index: 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); }
        25% { transform: translate(40px, -30px) rotate(90deg) scale(1.1); }
        50% { transform: translate(-20px, 40px) rotate(180deg) scale(0.9); }
        75% { transform: translate(30px, 20px) rotate(270deg) scale(1.05); }
    }
    
    @keyframes float-reverse {
        0%, 100% { transform: translate(0, 0) rotate(360deg) scale(1); }
        25% { transform: translate(-30px, 40px) rotate(270deg) scale(0.95); }
        50% { transform: translate(25px, -35px) rotate(180deg) scale(1.1); }
        75% { transform: translate(-40px, -20px) rotate(90deg) scale(0.9); }
    }
    
    .welcome-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        position: relative;
        z-index: 2;
    }
    .app-logo {
        font-size: 4rem;
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: pulse 2s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.05); }
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .app-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .login-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* 输入框样式优化 */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: #333;
        font-size: 1rem;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        background-color: rgba(255, 255, 255, 1);
    }
    
    /* 按钮样式优化 */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    .login-form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .demo-info {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .demo-info h4 {
        color: #ffffff !important;
        margin-bottom: 1rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .demo-info p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .register-link {
        text-align: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* 容器整体样式 */
    [data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[data-testid="stContainer"] {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        padding: 2rem;
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
        z-index: 10;
    }
    
    [data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[data-testid="stContainer"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        z-index: -1;
    }
    
    /* 确保所有内容在背景之上 */
    .welcome-header {
        position: relative;
        z-index: 15;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 添加明显的视觉光圈效果
    st.markdown("""
    <!-- 大型光圈装饰 -->
    <div style="position: fixed; top: 5%; left: 5%; width: 300px; height: 300px; 
                background: radial-gradient(circle, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0.2) 30%, transparent 70%); 
                border-radius: 50%; animation: mega-pulse 6s ease-in-out infinite; z-index: 5; pointer-events: none;"></div>
    <div style="position: fixed; bottom: 10%; right: 10%; width: 250px; height: 250px; 
                background: radial-gradient(circle, rgba(102,126,234,0.6) 0%, rgba(102,126,234,0.3) 25%, transparent 65%); 
                border-radius: 50%; animation: mega-pulse 4s ease-in-out infinite 2s; z-index: 5; pointer-events: none;"></div>
    <div style="position: fixed; top: 50%; left: 80%; width: 180px; height: 180px; 
                background: radial-gradient(circle, rgba(118,75,162,0.5) 0%, rgba(118,75,162,0.2) 35%, transparent 70%); 
                border-radius: 50%; animation: mega-pulse 5s ease-in-out infinite 1s; z-index: 5; pointer-events: none;"></div>
    <style>
    @keyframes mega-pulse {
        0%, 100% { transform: scale(0.8) rotate(0deg); opacity: 0.9; }
        50% { transform: scale(1.3) rotate(180deg); opacity: 0.4; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 应用介绍头部
    st.markdown("""
    <div class="welcome-header">
        <div class="app-logo">🎓</div>
        <div class="app-title">AlphaAgent</div>
        <div class="app-subtitle">LLM-Driven Quantitative Investment Research Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 使用streamlit的原生容器
        with st.container(border=True):
            st.markdown('<div class="login-title">User Login</div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Please enter username")
            password = st.text_input("Password", type="password", placeholder="Please enter password")
            
            col_login, col_clear = st.columns([1, 1])
            with col_login:
                login_clicked = st.form_submit_button("🚀 Login", use_container_width=True)
            with col_clear:
                clear_clicked = st.form_submit_button("🗑️ Clear", use_container_width=True)
            
            if login_clicked:
                if username and password:
                    if verify_password(username, password):
                        state.authenticated = True
                        state.username = username
                        st.success(f"Welcome back, {username}! 🎉")
                        st.rerun()
                    else:
                        st.error("Invalid username or password! ❌")
                else:
                    st.warning("Please enter username and password! ⚠️")
            
            if clear_clicked:
                st.rerun()
        
        # 注册链接
        st.markdown('<div class="register-link">', unsafe_allow_html=True)
        if st.button("📝 Don't have an account? Click to register", use_container_width=True):
            state.show_register = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 演示账户信息
        st.markdown("""
        <div class="demo-info">
            <h4>🔍 Demo Accounts</h4>
            <p><strong>Demo:</strong> demo / demo123</p>
        </div>
        """, unsafe_allow_html=True)

def logout():
    """登出功能"""
    state.authenticated = False
    state.username = None
    st.rerun()

# 初始化认证状态
if "authenticated" not in state:
    state.authenticated = False
if "username" not in state:
    state.username = None
if "show_register" not in state:
    state.show_register = False

# 检查认证状态
if not state.authenticated:
    if state.show_register:
        register_page()
    else:
        login_page()
    st.stop()



# 添加CSS样式
st.markdown("""
<style>
.metric-card {
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    background-color: transparent;
}
.metric-card:hover {
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}
.metric-title {
    color: #1f77b4;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
.plotly-chart {
    width: 100%;
    height: 100%;
}
.ideas-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: 10px 0;
}
.idea-card {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
}
.idea-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}
.idea-title {
    color: #1f77b4;
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 8px;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 4px;
    text-align: center;
}
.idea-content {
    font-size: 0.95em;
    color: inherit;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 10px 5px;
}
[data-testid="column"] {
    min-height: 250px;
    display: flex;
    flex-direction: column;
}
[data-testid="column"] > div {
    height: 100%;
}

/* Factor Agent 样式 */
.factor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 10px 0;
}
.factor-card {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.factor-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}
.factor-name {
    color: #2ca02c;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
    border-bottom: 2px solid #2ca02c;
    padding-bottom: 5px;
}
.factor-section {
    margin-bottom: 15px;
}
.factor-section-title {
    color: #1f77b4;
    font-size: 1em;
    font-weight: bold;
    margin-bottom: 5px;
}
.factor-section-content {
    background-color: rgba(0, 0, 0, 0.05);
    border-radius: 5px;
    padding: 10px;
    font-size: 0.95em;
    color: inherit;
}
.factor-expression {
    font-family: 'Courier New', Courier, monospace;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Factor Agent 新样式 */
.factor-container {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(49, 51, 63, 0.2);
    transition: all 0.3s ease;
}

.factor-container:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.factor-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(49, 51, 63, 0.2);
}

.factor-name {
    color: #2ca02c;
    font-size: 1.2em;
    font-weight: bold;
    flex-grow: 1;
}

.factor-content {
    background-color: rgba(49, 51, 63, 0.05);
    border-radius: 5px;
    padding: 15px;
    margin-top: 10px;
}

.factor-label {
    color: #1f77b4;
    font-size: 1em;
    font-weight: bold;
    margin-bottom: 5px;
}

/* 自定义代码块样式 */
.factor-code {
    background-color: rgba(0, 0, 0, 0.05);
    border-radius: 5px;
    padding: 10px;
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
    word-break: break-word;
    border: 1px solid rgba(49, 51, 63, 0.1);
}
</style>
""", unsafe_allow_html=True)

# 在文件开头添加
if '_watch' not in state:
    state._watch = True
    st.cache_data.clear()  # 使用 st.cache_data 替代 st.experimental_memo

# 在与state.current_task相关定义的地方附近添加自动刷新的state变量
if "current_task" not in state:
    state.current_task = None
if "api_base" not in state:
    state.api_base = "http://127.0.0.1:6701"  # 根据实际后端地址配置

# 获取log_path参数
parser = argparse.ArgumentParser(description="AlphaAgent Streamlit App")
parser.add_argument("--log_dir", required=True, type=str, help="Path to the log directory")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"Log dir `{main_log_path}` does not exist!")
        st.stop()
else:
    main_log_path = None


QLIB_SELECTED_METRICS = [
    "IC",
    "annualized_return",
    "information_ratio",
    "max_drawdown",
]

SIMILAR_SCENARIOS = (QlibAlphaAgentScenario, QlibModelScenario, QlibModelScenario, QlibFactorScenario, QlibFactorFromReportScenario)


def filter_log_folders(main_log_path):
    """
    The webpage only displays valid folders.
    If the __session__ folder exists in a subfolder of the log folder, it is considered a valid folder,
    otherwise it is considered an invalid folder.
    """
    folders = [
        folder.relative_to(main_log_path)
        for folder in main_log_path.iterdir()
        if folder.is_dir() and folder.joinpath("__session__").exists() and folder.joinpath("__session__").is_dir()
    ]
    # folders = sorted(folders, key=lambda x: x.name)
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(main_log_path, f)), reverse=True)
    return folders


if "log_path" not in state:
    if main_log_path:
        folders = filter_log_folders(main_log_path)
        state.log_path = folders[0] if folders else None  # 自动选择第一个（最新）
    else:
        state.log_path = None

if "scenario" not in state:
    state.scenario = None

if "fs" not in state:
    state.fs = None

if "msgs" not in state:
    state.msgs = defaultdict(lambda: defaultdict(list))

if "last_msg" not in state:
    state.last_msg = None

if "current_tags" not in state:
    state.current_tags = []

if "lround" not in state:
    state.lround = 0  # RD Loop Round

if "times" not in state:
    state.times = defaultdict(lambda: defaultdict(list))

if "erounds" not in state:
    state.erounds = defaultdict(int)  # Evolving Rounds in each RD Loop

if "e_decisions" not in state:
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))

# Summary Info
if "hypotheses" not in state:
    # Hypotheses in each RD Loop
    state.hypotheses = defaultdict(None)

if "h_decisions" not in state:
    state.h_decisions = defaultdict(bool)

if "metric_series" not in state:
    state.metric_series = []

# Factor Task Baseline
if "alpha158_metrics" not in state:
    state.alpha158_metrics = None

if "excluded_tags" not in state:
    state.excluded_tags = ["llm_messages"]  # 默认值
    
if "excluded_types" not in state:
    state.excluded_types = ["str"]  # 默认值

def should_display(msg: Message):
    for t in state.excluded_tags:
        if t in msg.tag.split("."):
            return False

    if type(msg.content).__name__ in state.excluded_types:
        return False

    return True


def get_msgs_until(end_func: Callable[[Message], bool] = lambda _: True):
    if state.fs:
        while True:
            try:
                msg = next(state.fs)
                if should_display(msg):
                    tags = msg.tag.split(".")
                    if "r" not in state.current_tags and "r" in tags:
                        state.lround += 1
                    if "evolving code" not in state.current_tags and "evolving code" in tags:
                        state.erounds[state.lround] += 1

                    state.current_tags = tags
                    state.last_msg = msg

                    # Update Summary Info
                    if "model runner result" in tags or "factor runner result" in tags or "runner result" in tags:
                        # factor baseline exp metrics
                        if isinstance(state.scenario, QlibFactorScenario) and state.alpha158_metrics is None:
                            sms = msg.content.based_experiments[0].result.loc[QLIB_SELECTED_METRICS]
                            sms.name = "alpha158"
                            state.alpha158_metrics = sms

                        if (
                            state.lround == 1
                            and len(msg.content.based_experiments) > 0
                            and msg.content.based_experiments[-1].result is not None
                        ):
                            sms = msg.content.based_experiments[-1].result
                            if isinstance(
                                state.scenario, (QlibModelScenario, QlibFactorFromReportScenario, QlibFactorScenario)
                            ):
                                sms = sms.loc[QLIB_SELECTED_METRICS]
                            sms.name = f"Baseline"
                            state.metric_series.append(sms)

                        # common metrics
                        if msg.content.result is not None:
                            sms = msg.content.result
                            if isinstance(
                                state.scenario, (QlibModelScenario, QlibFactorFromReportScenario, QlibFactorScenario)
                            ):
                                sms = sms.loc[QLIB_SELECTED_METRICS]

                            sms.name = f"Round {state.lround}"
                            state.metric_series.append(sms)
                    elif "hypothesis generation" in tags:
                        state.hypotheses[state.lround] = msg.content
                    elif "ef" in tags and "feedback" in tags:
                        state.h_decisions[state.lround] = msg.content.decision
                    elif "d" in tags:
                        if "evolving code" in tags:
                            msg.content = [i for i in msg.content if i]
                        if "evolving feedback" in tags:
                            total_len = len(msg.content)
                            msg.content = [i for i in msg.content if i]
                            none_num = total_len - len(msg.content)
                            if len(msg.content) != len(state.msgs[state.lround]["d.evolving code"][-1].content):
                                st.toast(":red[**Evolving Feedback Length Error!**]", icon="‼️")
                            right_num = 0
                            for wsf in msg.content:
                                if wsf.final_decision:
                                    right_num += 1
                            wrong_num = len(msg.content) - right_num
                            state.e_decisions[state.lround][state.erounds[state.lround]] = (
                                right_num,
                                wrong_num,
                                none_num,
                            )

                    state.msgs[state.lround][msg.tag].append(msg)

                    # Update Times
                    if "init" in tags:
                        state.times[state.lround]["init"].append(msg.timestamp)
                    if "r" in tags:
                        state.times[state.lround]["r"].append(msg.timestamp)
                    if "d" in tags:
                        state.times[state.lround]["d"].append(msg.timestamp)
                    if "ef" in tags:
                        state.times[state.lround]["ef"].append(msg.timestamp)

                    # Stop Getting Logs
                    if end_func(msg):
                        break
            except StopIteration:
                st.toast(":red[**No More Logs to Show!**]", icon="🛑")
                break


def refresh(same_trace: bool = False):
    if state.log_path is None:
        st.toast(":red[**Please Set Log Path!**]", icon="⚠️")
        return

    if main_log_path:
        state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    else:
        state.fs = FileStorage(state.log_path).iter_msg()

    # detect scenario
    if not same_trace:
        get_msgs_until(lambda m: not isinstance(m.content, str))
        if state.last_msg is None or not isinstance(state.last_msg.content, Scenario):
            st.toast(":red[**No Scenario Info detected**]", icon="❗")
            state.scenario = None
        else:
            state.scenario = state.last_msg.content
            st.toast(f":green[**Scenario Info detected**] *{type(state.scenario).__name__}*", icon="✅")

    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.last_msg = None
    state.current_tags = []
    state.alpha158_metrics = None
    state.times = defaultdict(lambda: defaultdict(list))
    
    if state.log_path is None:
        st.toast(":red[**Please Set Log Path!**]", icon="⚠️")
        return


def evolving_feedback_window(wsf: FactorSingleFeedback | ModelSingleFeedback):
    if isinstance(wsf, FactorSingleFeedback):
        ffc, efc, cfc, vfc = st.tabs(
            ["**Final Feedback🏁**", "Execution Feedback🖥️", "Code Feedback📄", "Value Feedback🔢"]
        )
        with ffc:
            st.code(wsf.final_feedback, language="log")
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.code(wsf.code_feedback, language="log")
        with vfc:
            st.code(wsf.value_feedback, language="log")
            
    elif isinstance(wsf, ModelSingleFeedback):
        ffc, efc, cfc, msfc, vfc = st.tabs(
            [
                "**Final Feedback🏁**",
                "Execution Feedback🖥️",
                "Code Feedback📄",
                "Model Shape Feedback📐",
                "Value Feedback🔢",
            ]
        )
        with ffc:
            st.markdown(wsf.final_feedback)
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.markdown(wsf.code_feedback)
        with msfc:
            st.markdown(wsf.shape_feedback)
        with vfc:
            st.markdown(wsf.value_feedback)



def display_hypotheses(hypotheses: dict[int, Hypothesis], decisions: dict[int, bool], round: int = None):
    if round is not None:
        hypotheses = {round: hypotheses.get(round)}
        decisions = {round: decisions.get(round)}
    
    name_dict = {
        "hypothesis": "RD-Agent proposes the hypothesis⬇️",
        "concise_justification": "because the reason⬇️",
        "concise_observation": "based on the observation⬇️",
        "concise_knowledge": "Knowledge⬇️ gained after practice",
    }
    
    # if success_only:
    #     shd = {k: v.__dict__ for k, v in hypotheses.items() if decisions[k]}
    # else:
    shd = {k: v.__dict__ for k, v in hypotheses.items()}
    
    df = pd.DataFrame(shd).T
    
    if "concise_observation" in df.columns and "concise_justification" in df.columns:
        df["concise_observation"], df["concise_justification"] = df["concise_justification"], df["concise_observation"]
        df.rename(
            columns={"concise_observation": "concise_justification", "concise_justification": "concise_observation"},
            inplace=True,
        )
    
    if "reason" in df.columns:
        df.drop(["reason"], axis=1, inplace=True)
    
    if "concise_reason" in df.columns:
        df.drop(["concise_reason"], axis=1, inplace=True)
    
    df.columns = df.columns.map(lambda x: name_dict.get(x, x))
    
    def style_rows(row):
        if decisions[row.name]:
            return ["color: green;"] * len(row)
        return [""] * len(row)
    
    def style_columns(col):
        if col.name != name_dict.get("hypothesis", "hypothesis"):
            return ["font-style: italic;"] * len(col)
        return ["font-weight: bold;"] * len(col)
    
    st.markdown(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0).to_html(), unsafe_allow_html=True)

# def display_hypotheses(hypotheses: dict[int, Hypothesis], decisions: dict[int, bool], success_only: bool = False):
#     name_dict = {
#         "hypothesis": "RD-Agent proposes the hypothesis⬇️",
#         "concise_justification": "because the reason⬇️",
#         "concise_observation": "based on the observation⬇️",
#         "concise_knowledge": "Knowledge⬇️ gained after practice",
#     }
#     if success_only:
#         shd = {k: v.__dict__ for k, v in hypotheses.items() if decisions[k]}
#     else:
#         shd = {k: v.__dict__ for k, v in hypotheses.items()}
#     df = pd.DataFrame(shd).T

#     if "concise_observation" in df.columns and "concise_justification" in df.columns:
#         df["concise_observation"], df["concise_justification"] = df["concise_justification"], df["concise_observation"]
#         df.rename(
#             columns={"concise_observation": "concise_justification", "concise_justification": "concise_observation"},
#             inplace=True,
#         )
#     if "reason" in df.columns:
#         df.drop(["reason"], axis=1, inplace=True)
#     if "concise_reason" in df.columns:
#         df.drop(["concise_reason"], axis=1, inplace=True)

#     df.columns = df.columns.map(lambda x: name_dict.get(x, x))

#     def style_rows(row):
#         if decisions[row.name]:
#             return ["color: green;"] * len(row)
#         return [""] * len(row)

#     def style_columns(col):
#         if col.name != name_dict.get("hypothesis", "hypothesis"):
#             return ["font-style: italic;"] * len(col)
#         return ["font-weight: bold;"] * len(col)

#     # st.dataframe(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0))
#     st.markdown(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0).to_html(), unsafe_allow_html=True)


def metrics_window(df: pd.DataFrame, R: int, C: int, *, height: int = 300, colors: list[str] = None):
    if len(df.columns) > R*C and R*C <= 8:
        df = df[[
            'IC', 'ICIR', 'Rank IC', 'Rank ICIR', 
            '1day.excess_return_with_cost.mean',
            '1day.excess_return_with_cost.annualized_return', 
            '1day.excess_return_with_cost.information_ratio', 
            '1day.excess_return_with_cost.max_drawdown'
                 ][:R*C]]
    
    # 去掉前缀
    df.columns = df.columns.str.replace('1day.excess_return_without_cost.', '')
    df.columns = df.columns.str.replace('1day.excess_return_with_cost.', '')
    
    # 创建子图
    fig = make_subplots(rows=R, cols=C, subplot_titles=df.columns)

    def hypothesis_hover_text(h: Hypothesis, d: bool = False):
        color = "green" if d else "black"
        text = h.hypothesis
        lines = textwrap.wrap(text, width=60)
        return f"<span style='color: {color};'>{'<br>'.join(lines)}</span>"
    
    hover_texts = [
        hypothesis_hover_text(state.hypotheses[int(i[6:])], state.h_decisions[int(i[6:])])
        for i in df.index[2:]
        if (i != "alpha158" and i.startswith('Round '))
    ]
    if state.alpha158_metrics is not None:
        hover_texts = ["Baseline: alpha158"] + hover_texts

    # 使用自定义颜色
    custom_colors = colors if colors else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for ci, col in enumerate(df.columns):
        row = ci // C + 1
        col_num = ci % C + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines+markers",
                connectgaps=True,
                marker=dict(
                    size=10, 
                    color=custom_colors[col_num-1],
                    line=dict(width=2, color='white')
                ),
                line=dict(width=3),
            ),
            row=row,
            col=col_num,
        )

    # 更新布局
    fig.update_layout(
        showlegend=False,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # 更新所有子图的样式
    for i in range(1, R + 1):
        for j in range(1, C + 1):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickvals=[df.index[0]] + list(df.index[1:]),
                ticktext=[f'<span style="color:#ff7f0e; font-weight:bold">{df.index[0]}</span>'] + list(df.index[1:]),
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                row=i,
                col=j,
            )

    # 使用卡片容器显示图表
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    # st.markdown('<div class="metric-title">Performance Metrics</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


def summary_window():
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("Runing Summary📊", divider="rainbow", anchor="_summary")
        if state.lround == 0:
            return
        with st.container():
            # TODO: not fixed height
            with st.container():
                bc, cc = st.columns([2, 2], vertical_alignment="center")
                with bc:
                    st.subheader("Metrics📈", anchor="_metrics")
                # with cc:
                #     show_true_only = st.toggle("successful hypotheses", value=False)

            # hypotheses_c, chart_c = st.columns([2, 3])
            chart_c = st.container()
            hypotheses_c = st.container()

            # with hypotheses_c:
            #     st.subheader("Hypotheses🏅", anchor="_hypotheses")
            #     display_hypotheses(state.hypotheses, state.h_decisions, show_true_only)

            with chart_c:
                if isinstance(state.scenario, QlibFactorScenario) and state.alpha158_metrics is not None:
                    df = pd.DataFrame([state.alpha158_metrics] + state.metric_series)
                else:
                    df = pd.DataFrame(state.metric_series)
                # if show_true_only and len(state.hypotheses) >= len(state.metric_series):
                #     if state.alpha158_metrics is not None:
                #         selected = ["alpha158"] + [i for i in df.index[2:] if state.h_decisions[int(i[6:])]]
                #     else:
                #         selected = [i for i in df.index if i == "Baseline" or state.h_decisions[int(i[6:])]]
                #     df = df.loc[selected]
                if df.shape[0] == 1:
                    st.table(df.iloc[0])
                elif df.shape[0] > 1:
                    if df.shape[1] == 1:
                        fig = px.line(df, x=df.index, y=df.columns, markers=True)
                        fig.update_layout(xaxis_title="Loop Round", yaxis_title=None)
                        st.plotly_chart(fig)
                    else:
                        metrics_window(df, 2, 4, height=600, colors=["red", "blue", "orange", "green"])



def tabs_hint():
    st.markdown(
        "<p style='font-size: small; color: #888888;'>You can navigate through the tabs using ⬅️ ➡️ or by holding Shift and scrolling with the mouse wheel🖱️.</p>",
        unsafe_allow_html=True,
    )


def tasks_window(tasks: list[FactorTask | ModelTask]):
    if isinstance(tasks[0], FactorTask):
        title = "Factor Agent⚙️"
        st.subheader(title, divider="blue", anchor="_factor")
        
        for ft in tasks:
            # 使用 Streamlit 容器创建卡片效果
            with st.container():
                # 添加一些上下边距
                # st.markdown("<br>", unsafe_allow_html=True)
                
                # 使用 expander 创建可展开的卡片
                with st.expander(f"### 🔍 **{ft.factor_name}**", expanded=True):
                    # Description 部分
                    st.markdown("##### Description")
                    st.code(ft.factor_description, language="plaintext")
                    
                    # Expression 部分
                    st.markdown("##### Expression")
                    # 使用 success 样式代替 info，显示为绿色背景
                    st.code(f"{ft.factor_expression}", language="python")
                
                # 添加分隔
                st.markdown("<br>", unsafe_allow_html=True)

    elif isinstance(tasks[0], ModelTask):
        st.markdown("**Model Tasks🚩**")
        tnames = [m.name for m in tasks]
        if sum(len(tn) for tn in tnames) > 100:
            tabs_hint()
        tabs = st.tabs(tnames)
        for i, mt in enumerate(tasks):
            with tabs[i]:
                st.markdown(f"**Model Type**: {mt.model_type}")
                st.markdown(f"**Description**: {mt.description}")
                st.latex("Formulation")
                st.latex(mt.formulation)

                mks = "| Variable | Description |\n| --- | --- |\n"
                if mt.variables:
                    for v, d in mt.variables.items():
                        mks += f"| ${v}$ | {d} |\n"
                    st.markdown(mks)


def research_window(round: int):
    with st.container(border=True):
        title = "Idea Agent💡"
        st.subheader(title, divider="blue", anchor="_idea")
        if isinstance(state.scenario, SIMILAR_SCENARIOS):
            # pdf image
            if pim := state.msgs[round]["r.extract_factors_and_implement.load_pdf_screenshot"]:
                for i in range(min(2, len(pim))):
                    st.image(pim[i].content, use_container_width=True)

            # Hypothesis
            if hg := state.msgs[round]["r.hypothesis generation"]:
                h: Hypothesis = hg[0].content
                
                # 创建网格布局的HTML
                cards_html = f"""
                <div class="ideas-grid">
                    <div class="idea-card">
                        <div class="idea-title">Hypothesis</div>
                        <div class="idea-content">{h.hypothesis}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">Justification</div>
                        <div class="idea-content">{h.concise_justification}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">Knowledge</div>
                        <div class="idea-content">{h.concise_knowledge}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">Specification</div>
                        <div class="idea-content">By combining Intraday Price Velocity with volume and volatility data within a specific time window and analyzing their collective impact on short-term returns, we aim to enhance the model's predictive power and capture a more nuanced understanding of market dynamics, thereby increasing the accuracy of short-term return predictions.</div>
                    </div>
                </div>
                """
                
                st.markdown(cards_html, unsafe_allow_html=True)

            if eg := state.msgs[round]["r.experiment generation"]:
                tasks_window(eg[0].content)



def feedback_window():
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        with st.container(border=True):
            st.subheader("Eval Agent📝", divider="orange", anchor="_eval")

            if state.lround > 0 and isinstance(
                state.scenario, (QlibModelScenario, QlibFactorScenario, QlibFactorFromReportScenario)
            ):
                with st.expander("**Config**", expanded=True):
                    st.markdown(state.scenario.experiment_setting, unsafe_allow_html=True)
            
            if fbr := state.msgs[round]["ef.Quantitative Backtesting Chart"]:
                # st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("#### PnL Figure📈")
                num_fig = len(state.msgs[round]["ef.Quantitative Backtesting Chart"])
                if num_fig > 1:
                    for i in range(num_fig):
                        if i == 0:
                            # 使用 HTML 实现居中
                            st.markdown(
                                "<div style='text-align: center;'><strong>Baseline</strong></div>", 
                                unsafe_allow_html=True
                            )
                        fig = report_figure(fbr[i].content)
                        st.plotly_chart(fig)
                        if i < num_fig - 1:  # 在图表之间添加分割线
                            st.divider()
                else:
                    fig = report_figure(fbr[0].content)
                    st.plotly_chart(fig)
            if fbn := state.msgs[round]["ef.runner result"]:
                # 添加空行
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("#### Runner Result Backtesting Table 📌")
                # 获取结果数据
                runner_result_data = fbn[0].content
                result = runner_result_data.result
                # 将结果转化为 DataFrame
                result_df = pd.DataFrame(result) if isinstance(result, pd.Series) else pd.DataFrame(result)
                result_df = result_df.reset_index()
                result_df.columns = ["Metric", "Value"]
                
                # 添加Category列来分类指标
                def categorize_metric(metric):
                    if "without_cost" in metric:
                        return "Without Cost"
                    elif "with_cost" in metric:
                        return "With Cost"
                    else:
                        return "Other Metrics"
                
                result_df['Category'] = result_df['Metric'].apply(categorize_metric)
                
                # 清理Metric名称
                result_df['Metric'] = result_df['Metric'].apply(lambda x: x.split('.')[-1].replace('_', ' ').title())
                
                # 规范化指标名称
                metric_name_map = {
                    'Ic': 'IC',
                    'Icir': 'ICIR',
                    'Rank Ic': 'Rank IC',
                    'Rank Icir': 'Rank ICIR',
                    'Ffr': 'ffr',
                    'Pa': 'pa',
                    'Pos': 'pos'
                }
                result_df['Metric'] = result_df['Metric'].apply(lambda x: metric_name_map.get(x, x))
                
                # 设置表格样式
                st.markdown("""
                <style>
                .metric-table {
                    font-size: 1em;
                    border-collapse: collapse;
                    margin: 25px 0;
                    width: 100%;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    overflow: hidden;
                }
                .metric-table thead tr {
                    background-color: #1f77b4;
                    color: white;
                    text-align: left;
                    font-weight: bold;
                }
                .metric-table th,
                .metric-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                .metric-table tbody tr {
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }
                .metric-table tbody tr:nth-of-type(even) {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                .metric-table tbody tr:last-of-type {
                    border-bottom: 2px solid #1f77b4;
                }
                .category-header {
                    background-color: rgba(31, 119, 180, 0.1) !important;
                    font-weight: bold;
                    color: #1f77b4;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # 创建HTML表格
                table_html = '<table class="metric-table"><thead><tr><th>Category</th><th>Metric</th><th>Value</th></tr></thead><tbody>'
                
                # 按Category分组添加行
                for category in ['Without Cost', 'With Cost', 'Other Metrics']:
                    category_data = result_df[result_df['Category'] == category]
                    if not category_data.empty:
                        # 添加类别标题行
                        table_html += f'<tr class="category-header"><td colspan="3">{category}</td></tr>'
                        # 添加该类别的所有指标
                        for _, row in category_data.iterrows():
                            table_html += f'<tr><td></td><td>{row["Metric"]}</td><td>{row["Value"]:.4f}</td></tr>'
                
                table_html += '</tbody></table>'
                
                # 显示表格
                st.markdown(table_html, unsafe_allow_html=True)
            if fb := state.msgs[round]["ef.feedback"]:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("#### Hypothesis Feedback🔍")
                h: HypothesisFeedback = fb[0].content
                
                # 使用网格布局显示反馈内容
                feedback_html = """
                <div class="ideas-grid">
                    <div class="idea-card">
                        <div class="idea-title">Observations</div>
                        <div class="idea-content">{}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">Hypothesis Evaluation</div>
                        <div class="idea-content">{}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">New Hypothesis</div>
                        <div class="idea-content">{}</div>
                    </div>
                    <div class="idea-card">
                        <div class="idea-title">Decision & Reason</div>
                        <div class="idea-content">Decision: {}<br><br>Reason: {}</div>
                    </div>
                </div>
                """.format(
                    h.observations,
                    h.hypothesis_evaluation,
                    h.new_hypothesis,
                    h.decision,
                    h.reason
                )
                st.markdown(feedback_html, unsafe_allow_html=True)

            # if isinstance(state.scenario, KGScenario):
            #     if fbe := state.msgs[round]["ef.runner result"]:
            #         submission_path = fbe[0].content.experiment_workspace.workspace_path / "submission.csv"
            #         st.markdown(
            #             f":green[**Exp Workspace**]: {str(fbe[0].content.experiment_workspace.workspace_path.absolute())}"
            #         )
            #         try:
            #             data = submission_path.read_bytes()
            #             st.download_button(
            #                 label="**Download** submission.csv",
            #                 data=data,
            #                 file_name="submission.csv",
            #                 mime="text/csv",
            #             )
            #         except Exception as e:
            #             st.markdown(f":red[**Download Button Error**]: {e}")


def evolving_window():
    title = "Debugging" if isinstance(state.scenario, SIMILAR_SCENARIOS) else "Development🛠️ (evolving coder)"
    st.subheader(title, divider="green", anchor="_debugging")

    # Evolving Status
    if state.erounds[round] > 0:
        st.markdown("##### **☑️ Evolving Status**")
        es = state.e_decisions[round]
        e_status_mks = "".join(f"| {ei} " for ei in range(1, state.erounds[round] + 1)) + "|\n"
        e_status_mks += "|--" * state.erounds[round] + "|\n"
        for ei, estatus in es.items():
            if not estatus:
                estatus = (0, 0, 0)
            e_status_mks += "| " + "🕙<br>" * estatus[2] + "✔️<br>" * estatus[0] + "❌<br>" * estatus[1] + " "
        e_status_mks += "|\n"
        st.markdown(e_status_mks, unsafe_allow_html=True)

    # Evolving Tabs
    if state.erounds[round] > 0:
        if state.erounds[round] > 1:
            evolving_round = st.radio(
                "**🔄️Evolving Rounds**",
                horizontal=True,
                options=range(1, state.erounds[round] + 1),
                index=state.erounds[round] - 1,
                key="show_eround",
            )
        else:
            evolving_round = 1

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["d.evolving code"][
            evolving_round - 1
        ].content
        
        tab_names = [
            w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
        ]
        if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
            for j in range(len(ws)):
                if state.msgs[round]["d.evolving feedback"][evolving_round - 1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"
                    
        if sum(len(tn) for tn in tab_names) > 100:
            tabs_hint()
            
        wtabs = st.tabs(tab_names)
        for j, w in enumerate(ws):
            with wtabs[j]:
                # if 'file_dict' in w.__dict__:
                #     for k, v in w.file_dict.items():
                #         with st.expander(f":green[`{k}`]", expanded=True):
                #             st.code(v, language="python")
                # continue


                # Evolving Code
                st.markdown(f"**Workspace Path**: {w.workspace_path}")
                expr = re.search(r"expr\s*=\s*\"(.*?)\"", w.code_dict['factor.py'], re.DOTALL).group(1)
                # 只展示表达式而不是整个代码块
                expression = w.target_task.factor_expression
                st.markdown(f"- ##### **Expression** ✨: \n```\n{expr}\n```")

                # Evolving Feedback
                if len(state.msgs[round]["d.evolving feedback"]) >= evolving_round:
                    evolving_feedback_window(state.msgs[round]["d.evolving feedback"][evolving_round - 1].content[j])


## [Scenario Description](#_scenario)
toc = """
## [Summary📊](#_summary)
- [**Metrics📈**](#_metrics)
## [AlphaAgent Loops♾️](#_loops)
- [**Idea Agent💡**](#_idea)
- [**Factor Agent⚙️**](#_factor)
- [**Eval Agent📝**](#_eval)
"""
# Config Sidebar
with st.sidebar:
    st.markdown("# **AlphaAgent**✨")
    
    # 用户信息
    st.markdown("---")
    st.markdown(f"👤 **Current User:** {state.username}")
    if st.button("🚪 Logout", use_container_width=True):
        logout()
    st.markdown("---")
    
    st.subheader(":blue[Table of Content]", divider="blue")
    st.markdown(toc)
    st.subheader(":blue[Control Panel]", divider="blue")



    with st.container(border=True):
        if main_log_path:
            lc1, lc2 = st.columns([1, 2], vertical_alignment="center")
            with lc1:
                st.markdown(":blue[**Log Path**]")
            with lc2:
                manually = st.toggle("Manual Input")
            if manually:
                st.text_input("log path", key="log_path", on_change=refresh, label_visibility="collapsed")
            else:
                folders = filter_log_folders(main_log_path)
                # 按修改时间排序，最新的在最前面
                st.selectbox(f"**Select from `{main_log_path}`**", folders, key="log_path", on_change=refresh)
        else:
            st.text_input(":blue[**log path**]", key="log_path", on_change=refresh)

    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        if st.button(":green[**All Loops**]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: False)
        if st.button("**Reset**", use_container_width=True):
            refresh(same_trace=True)
    with c2:
        if st.button(":green[Next Loop]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "ef.feedback" in m.tag)

        if st.button("Next Step", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "d.evolving feedback" in m.tag)

    with st.popover(":orange[**Config⚙️**]", use_container_width=True):
        st.multiselect("excluded log tags", ["llm_messages"], ["llm_messages"], key="excluded_tags")
        st.multiselect("excluded log types", ["str", "dict", "list"], ["str"], key="excluded_types")

    if args.debug:
        debug = st.toggle("debug", value=False)

        if debug:
            if st.button("Single Step Run", use_container_width=True):
                get_msgs_until()
    else:
        debug = False
        
    
    st.subheader(":blue[Entrance]", divider="blue")
    user_hypothesis = st.text_input("🔍 **Enter an hypothesis you want to verify**",
        value=state.get("user_direction", ""),
        placeholder="..."
    )
    
    # 启动/停止按钮
    col1, col2 = st.columns([1, 1])
    with col1:
        start_clicked = st.button(
            "🚀 Start Mining" if not state.current_task else "⏳ Mining...",
            disabled=state.current_task is not None,
            use_container_width=True
        )
    with col2:
        stop_clicked = st.button(
            "⏹ Stop Mining",
            disabled=state.current_task is None,
            use_container_width=True
        )
    
    # 处理按钮点击事件
    if start_clicked and user_hypothesis:
        response = requests.post(
            f"{state.api_base}/api/tasks",
            json={"direction": user_hypothesis}
        )
        if response.status_code == 200:
            state.current_task = response.json()["task_id"]
            state.user_direction = user_hypothesis
        refresh(same_trace=True)
        st.rerun()
    
    if stop_clicked and state.current_task:
        print("Stop posted")
        response = requests.post(
            f"{state.api_base}/api/tasks/{state.current_task}/stop"
        )
        
        if response.status_code == 200:
            st.success("Stop signal sent")
            state.current_task = None
            print("Stop succeeds")
        st.rerun()

    # 删除自动刷新控制代码，仅保留手动刷新按钮
    if state.current_task:
        # 手动刷新按钮 - 使用英文
        if st.button("🔄 Refresh Now", use_container_width=True):
            refresh(same_trace=True)
            get_msgs_until(lambda m: False)
            st.rerun()


# Debug Info Window
if debug:
    with st.expander(":red[**Debug Info**]", expanded=True):
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            st.markdown(
                f"**log path**: {state.log_path}\n\n"
                f"**excluded tags**: {state.excluded_tags}\n\n"
                f"**excluded types**: {state.excluded_types}\n\n"
                f":blue[**message id**]: {sum(sum(len(tmsgs) for tmsgs in rmsgs.values()) for rmsgs in state.msgs.values())}\n\n"
                f":blue[**round**]: {state.lround}\n\n"
                f":blue[**evolving round**]: {state.erounds[state.lround]}\n\n"
            )
        with dcol2:
            if state.last_msg:
                st.write(state.last_msg)
                if isinstance(state.last_msg.content, list):
                    st.write(state.last_msg.content[0])
                elif not isinstance(state.last_msg.content, str):
                    st.write(state.last_msg.content.__dict__)


if state.log_path and state.fs is None:
    refresh()

# Main Window
# header_c1, header_c3 = st.columns([1, 6], vertical_alignment="center")
# with st.container():
#     with header_c1:
#         st.image("https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31")
#     with header_c3:
#         st.markdown(
#             """
#         <h1>
#             RD-Agent:<br>LLM-based autonomous evolving agents for industrial data-driven R&D
#         </h1>
#         """,
#             unsafe_allow_html=True,
#         )

# Project Info
# with st.container():
#     image_c, scen_c = st.columns([3, 3], vertical_alignment="center")
#     with image_c:
#         img_path = rfiles("rdagent.log.ui").joinpath("flow.png")
#         st.image(str(img_path), use_container_width=True)
#     with scen_c:
#         st.header("Scenario Description📖", divider="violet", anchor="_scenario")
#         if state.scenario is not None:
#             theme = st_theme()
#             if theme:
#                 theme = theme.get("base", "light")
#             css = f"""
# <style>
#     a[href="#_rdloops"], a[href="#_research"], a[href="#_development"], a[href="#_feedback"], a[href="#_scenario"], a[href="#_summary"], a[href="#_hypotheses"], a[href="#_metrics"] {{
#         color: {"black" if theme == "light" else "white"};
#     }}
# </style>
# """
#             st.markdown(state.scenario.rich_style_description + css, unsafe_allow_html=True)


def show_times(round: int):
    for k, v in state.times[round].items():
        if len(v) > 1:
            diff = v[-1] - v[0]
        else:
            diff = v[0] - v[0]
        total_seconds = diff.seconds
        seconds = total_seconds % 60
        minutes = total_seconds // 60
        st.markdown(f"**:blue[{k}]**: :red[**{minutes}**] minutes :orange[**{seconds}**] seconds")


if state.scenario is not None:
    summary_window()

    # R&D Loops Window
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("AlphaAgent Loops♾️", divider="rainbow", anchor="_loops")
        # st.markdown("#### Loops")
        if len(state.msgs) > 1:
            r_options = list(state.msgs.keys())
            if 0 in r_options:
                r_options.remove(0)
            round = st.radio("# **Loop**", horizontal=True, options=r_options, index=state.lround - 1)
        else:
            round = 1

        # show_times(round)
        # rf_c, d_c = st.columns([2, 2])
        r_c = st.container()
        d_c = st.container()
        f_c = st.container()
    else:
        st.error("Unknown Scenario!")
        st.stop()

    with r_c:
        research_window(round)
    with f_c:
        feedback_window()

    with d_c.container(border=True):
        evolving_window()


st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("#### Disclaimer")
st.markdown(
    "*This content is AI-generated and may not be fully accurate or up-to-date; please verify with a professional for critical matters.*",
    unsafe_allow_html=True,
)
